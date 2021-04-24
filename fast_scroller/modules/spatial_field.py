import os
import time
from collections import Counter
from functools import partial

import numpy as np
import scipy.optimize as so

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from traits.api import Instance, Button, HasTraits, Float, Bool, Enum, on_trait_change, Int, File
from traitsui.api import View, VGroup, HGroup, Item, UItem, HSplit, VSplit, Label, RangeEditor, FileEditor

from tqdm import tqdm

from nitime.utils import autocov

from ecogdata.channel_map import ChannelMap
from ecogdata.filt.blocks import BlockedSignal
import ecoglib.estimation.spatial_variance as sv
from ecoglib.signal_testing import spatial_autocovariance
from ecoglib.estimation import cxx_to_pairs, matern_semivariogram, matern_spectrum
from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.vis.gui_tools import ArrayMap, SavesFigure
from ecoglib.vis.plot_util import subplots, subplot2grid
import seaborn as sns

from .base import PlotsInterval, MultiframeSavesFigure, colormaps, FigureCanvas
from ..helpers import PersistentWindow

__all__ = ['ArrayVarianceTool', 'SpatialVariance']


class ArrayVarianceTool(PersistentWindow):
    """Stand-alone UI to image covariance seeded at chosen sites"""
    array_plot = Instance(ArrayMap)
    selected_site = Int(-1)
    min_max_norm = Bool(True)
    _c_lo = Float
    _c_hi = Float
    cmap = Enum('gray', colormaps)
    save_file = File(os.getcwd())
    save_all = Button('Save PDF')

    def __init__(self, cov, chan_map, **traits):
        self.cov = cov
        self._cv_mn = cov.min()
        self._cv_mx = cov.max()
        array_plot = ArrayMap(chan_map, vec=self.cov.mean(1))
        HasTraits.__init__(
            self, array_plot=array_plot,
            _c_lo=self._cv_mn, _c_hi=self._cv_mx
        )
        self.sync_trait('selected_site', self.array_plot, mutual=True)

    @on_trait_change('selected_site, cmap, min_max_norm, _c_lo, _c_hi')
    def _image(self):
        if not hasattr(self, 'array_plot'):
            return
        if self.array_plot is None:
            return
        kw = dict(cmap=self.cmap)
        if self.min_max_norm:
            kw['clim'] = (self._cv_mn, self._cv_mx)
        else:
            kw['clim'] = self._c_lo, self._c_hi
        site = self.selected_site
        if site >= 0:
            self.array_plot.ax.set_title('Seeded covariance map')
            self.array_plot.update_map(self.cov[site], **kw)
        else:
            self.array_plot.ax.set_title('Mean covariance weight')
            self.array_plot.update_map(self.cov.mean(1), **kw)

    def _save_all_fired(self):
        # cycle through sites, update_zoom_callback image, and use
        # PdfPages.savefig(self.array_plot.fig)
        chan_map = self.array_plot.chan_map
        ij = list(zip(*chan_map.to_mat()))
        ij = sorted(ij)
        pdf_file = self.save_file
        if not pdf_file.endswith('.pdf'):
            pdf_file = pdf_file + '.pdf'
        save_site = self.selected_site
        with PdfPages(pdf_file) as pdf:
            for i_, j_ in tqdm(ij, desc='Saving PDF pages', leave=True):
                s = chan_map.lookup(i_, j_)
                self.selected_site = s
                f = self.array_plot.fig
                ax = self.array_plot.ax
                ttl = ax.get_title()
                ax.set_title(ttl + ' site ({0}, {1})'.format(i_, j_))
                pdf.savefig(f)
        self.selected_site = save_site

    def _post_canvas_hook(self):
        self.array_plot.fig.canvas.mpl_connect(
            'button_press_event', self.array_plot.click_listen
        )
        self._image()

    def default_traits_view(self):
        v = View(
            VSplit(
                UItem('array_plot', editor=MPLFigureEditor(),
                      width=500, height=400, resizable=True),
                HGroup(
                    Item('selected_site', style='readonly'),
                    Item('min_max_norm', label='Min-Max Normalize'),
                    Item('cmap', label='Color map'),
                ),
                HGroup(
                    Item('_c_lo', label='Low color'),
                    Item('_c_hi', label='High color'),
                    enabled_when='min_max_norm==False'
                ),
                HGroup(
                    Item('save_file', label='pdf file',
                         editor=FileEditor(dialog_style='save')),
                    UItem('save_all')
                )
            ),
            handler=PingPongStartup,
            resizable=True,
            title='Covariance Visualization'
        )
        return v


def make_matern_label(**params):
    """Helper function for plot labels."""
    label = ''
    if 'nu' in params:
        label = label + '\u03BD {nu:.1f} '
    if 'theta' in params:
        label = label + '\u03B8 {theta:.1f} '
    if 'nugget' in params:
        label = label + '\u03C3 {nugget:.2f} '
    if 'sill' in params:
        label = label + '\u03C2 {sill:.2f} '

    return label.format(**params)


class STSemivar(PersistentWindow):

    sv_fig = Instance(Figure)
    array_frame = Instance(Figure)
    cnx = Instance(Figure)
    tsfig = Instance(Figure)

    _lo = Int(0)
    _hi = Int
    slider = Int(0)
    fitting = Bool(False)

    def __init__(self, x, t, st_semivar, timeseries, chan_map, **traits):
        self._hi = len(t) - 1
        self._x = x
        self._t = t
        self._timeseries = timeseries
        self._st_semivar = st_semivar
        self._chan_map = chan_map

        combs = chan_map.site_combinations
        self._binned = (len(x) != len(np.unique(combs.dist)))
        if self._binned:
            diffs = np.abs(combs.dist - x[:, None])
            self._bin_map = diffs.argmin(0)

        # set up semivariogram figure
        self.sv_fig = Figure(figsize=(5, 4))
        ax = self.sv_fig.add_subplot(111)
        self._sv_line = ax.plot(x, st_semivar[:, 0], marker='o', ls='--')[0]
        ax.set_ylim(0, st_semivar.max())
        sns.despine(ax=ax)
        self._marked_point = None
        self._fit_line = None
        self._info_text = None

        # set up raw potential map
        self.array_frame = Figure(figsize=(5, 4))
        ax = self.array_frame.add_subplot(111)
        frame = chan_map.embed(
            self._timeseries[:, int(self._t[self.slider])]
        )
        clim = np.percentile(self._timeseries, [2, 98])
        self._lfp_img = ax.imshow(frame, clim=clim, origin='upper')

        # set up raw timeseries plot
        self.tsfig = Figure(figsize=(5, 4))
        ax = self.tsfig.add_subplot(111)
        lines = [list(enumerate(ts)) for ts in timeseries]
        lc = LineCollection(lines, colors='k', alpha=0.5, linewidths=0.5)
        ax.add_collection(lc)
        ax.autoscale_view(True, True, True)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Voltage')
        self._time_marker = ax.axvline(t[self.slider], color='r', lw=1)

        # set up site-site connections plot
        self.cnx = Figure(figsize=(5, 5))
        ax = self.cnx.add_subplot(111)
        i, j = chan_map.to_mat()
        ax.scatter(j, i, s=10)
        ax.set_ylim(i.max() + 0.5, i.min() - 0.5)
        ax.axis('image')
        self._cnx_lines = None

        super(STSemivar, self).__init__(**traits)

    @staticmethod
    def from_array_and_lag(x, n, lag, chan_map, normed=False, **kwargs):
        """
        Create new STSemivar object from array and short-time specs.

        Parameters
        ----------

        x : ndarray
            n_site x n_samples
        n : int
            short-time length in samples
        lag : int
            lag (step) length in samples
        chan_map : ChannelMap
            site to array lookup
        normed : bool
            Normalize (unit) variance (default False)

        Other keyword arguments are passed through to the
        semivariogram estimator.
        """
        t = np.arange(x.shape[-1])

        xb = BlockedSignal(x.copy(), n, overlap=n - lag)
        tb = BlockedSignal(t, n, overlap=n - lag)

        st_svar = []
        block_time = []
        combs = chan_map.site_combinations

        for i in range(len(xb)):

            xb_ = xb.block(i)
            if normed:
                xb_ = xb_ / xb_.std(1)[:, None]

            sx, sy = sv.semivariogram(xb_, combs, **kwargs)
            st_svar.append(sy)
            block_time.append(tb.block(i).mean())

        st_svar = np.array(st_svar).T
        block_time = np.array(block_time)
        return STSemivar(sx, block_time, st_svar, x, chan_map)

    @on_trait_change('slider')
    def _slider_changed(self):
        # 1) semivar plot (and marked point) (and fit line)
        self._sv_line.set_data(self._x, self._st_semivar[:, self.slider])
        if self._marked_point is not None:
            mx, my = self._marked_point.get_data()
            xbin = np.argmin(np.abs(self._x - mx[0]))
            x = self._x[xbin]
            y = self._st_semivar[xbin, self.slider]
            self._marked_point.set_data([x], [y])

        if self.fitting:
            self._draw_fit()

        # 2) heatmap
        f = self._timeseries[:, int(self._t[self.slider])]
        m = self._chan_map.embed(f)
        self._lfp_img.set_array(m)
        # 3) time marker on timeseries
        self._time_marker.set_data(
            [self._t[self.slider], self._t[self.slider]], [0, 1]
        )
        for f in (self.sv_fig, self.array_frame, self.tsfig):
            f.canvas.draw_idle()

    @on_trait_change('fitting')
    def _toggle_fit(self):
        if self._fit_line is None:
            self._draw_fit()
        else:
            self._fit_line.remove()
            self._info_text.remove()
            self._fit_line = self._info_text = None

    def fit_semivar(self, xf=None):
        x, y = self._sv_line.get_data()
        yl = self._sv_line.axes.get_ylim()[1]
        bounds = {
            'theta': (0.5, x.max()),
            'nu': (0.25, 5),
            'nugget': (0, y.min()),
            'sill': (y.mean(), 1.5 * yl)
        }
        prm = matern_semivariogram(
            x, y=y, free=('theta', 'nu', 'sill', 'nugget'), bounds=bounds
        )
        if xf is None:
            xf = x
        y_est = matern_semivariogram(xf, **prm)
        return prm, y_est

    def _get_info_text(self, prm):
        matern_label = make_matern_label(**prm)
        theta = prm['theta']
        nu = prm['nu']
        s0 = matern_spectrum(0, theta=theta, nu=nu)

        def fn(x, t, n):
            return matern_spectrum(x, t, n) - s0 / 100
        kc = so.brentq(fn, 0, 3, args=(theta, nu))
        bw_label = 'Crit dens: {0:.2f} mm'.format((2 * kc) ** -1)
        label = '\n'.join([matern_label, bw_label])
        return label

    def _draw_fit(self):
        ax = self.sv_fig.axes[0]
        xl = ax.get_xlim()
        xf = np.linspace(xl[0], xl[1], 100)
        prm, y = self.fit_semivar(xf=xf)
        label = self._get_info_text(prm)
        if self._fit_line is None:
            self._fit_line = ax.plot(xf, y, color='r', ls='-')[0]
            self._info_text = ax.text(
                0.1, 0.75, label, fontsize=10, transform=ax.transAxes
            )
        else:
            self._fit_line.set_data(xf, y)
            self._info_text.set_text(label)

        if self.sv_fig.canvas:
            self.sv_fig.canvas.draw_idle()

    def _setup_picker(self):
        xscale = self.sv_fig.axes[0].transData.get_matrix()[0, 0]
        self._sv_line.set_picker(True)
        self._sv_line.set_pickradius(xscale * np.diff(self._x).min())
        self.sv_fig.canvas.mpl_connect('pick_event', self._pick_pairs)

    def _pick_pairs(self, event):
        m_event = event.mouseevent

        px = m_event.xdata
        xbin = np.argmin(np.abs(self._x - px))
        x = self._x[xbin]
        y = self._st_semivar[xbin, self.slider]

        # unmark previous marked point (if any)

        if self._marked_point is not None:
            mx, my = self._marked_point.get_data()
            self._marked_point.remove()
            self._cnx_lc.remove()
            pc = self.cnx.axes[0].collections[0]
            pc.set_sizes(np.ones(len(self._chan_map)) * 10)

            self._marked_point = self._cnx_lc = None
            # if the picked point was the old point, then return
            if np.abs(mx[0] - x) < 1e-5 and np.abs(my[0] - y) < 1e-5:
                self.sv_fig.canvas.draw_idle()
                self.cnx.canvas.draw_idle()
                return

        # 1) change marked point
        self._marked_point = self.sv_fig.axes[0].plot(
            [x], [y], marker='o', ls='', color='k'
        )[0]

        # 2) create connection lines
        combs = self._chan_map.site_combinations
        if self._binned:
            mask = self._bin_map == xbin
        else:
            mask = combs.dist == x
        lines = [(i2, i1)
                 for i1, i2 in zip(combs.idx1[mask], combs.idx2[mask])]
        self._cnx_lc = LineCollection(
            lines, linewidths=0.5, alpha=0.5, colors='r')
        self.cnx.axes[0].add_collection(self._cnx_lc)
        # 3) change site radius based on connectivity
        i1 = combs.idx1[mask]
        i2 = combs.idx2[mask]
        c = Counter(map(tuple, np.row_stack((i1, i2)).tolist()))
        sizes = [c.get(ij, 0) for ij in zip(*self._chan_map.to_mat())]
        sizes = np.array(sizes) - np.min(sizes)
        pc = self.cnx.axes[0].collections[0]
        pc.set_sizes(sizes * 5)

        self.cnx.canvas.draw_idle()
        self.sv_fig.canvas.draw_idle()

    def _post_canvas_hook(self):
        for f in (self.sv_fig, self.array_frame, self.tsfig, self.cnx):
            f.tight_layout(pad=0.5)
            f.canvas.draw_idle()
        from pyqtgraph.Qt import QtCore
        self.sv_fig.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.sv_fig.canvas.setFocus()
        self._setup_picker()

    def default_traits_view(self):
        fig = self.sv_fig
        sv_fh = int(fig.get_figheight() * fig.get_dpi())
        sv_fw = int(fig.get_figwidth() * fig.get_dpi())

        fig = self.array_frame
        af_fh = int(fig.get_figheight() * fig.get_dpi())
        af_fw = int(fig.get_figwidth() * fig.get_dpi())

        fig = self.cnx
        cn_fh = int(fig.get_figheight() * fig.get_dpi())
        cn_fw = int(fig.get_figwidth() * fig.get_dpi())

        fig = self.tsfig
        ts_fh = int(fig.get_figheight() * fig.get_dpi())
        ts_fw = int(fig.get_figwidth() * fig.get_dpi())

        traits_view = View(
            VSplit(
                HSplit(
                    UItem(
                        'sv_fig',
                        editor=MPLFigureEditor(),
                        resizable=True, height=sv_fh, width=sv_fw
                    ),
                    UItem(
                        'array_frame',
                        editor=MPLFigureEditor(),
                        resizable=True, height=af_fh, width=af_fw
                    )
                ),
                HSplit(
                    UItem(
                        'cnx',
                        editor=MPLFigureEditor(),
                        resizable=True, height=cn_fh, width=cn_fw
                    ),
                    UItem(
                        'tsfig',
                        editor=MPLFigureEditor(),
                        resizable=True, height=ts_fh, width=ts_fw
                    ),
                ),
                HGroup(
                    Item('fitting', label='Fit semivar'),
                    Item('slider', label='Scroll frames',
                         editor=RangeEditor(low=self._lo, high=self._hi)),
                )
            ),
            resizable=True,
            handler=PingPongStartup()
        )
        return traits_view


class SpatialVariance(PlotsInterval):
    name = 'Spatial Variance'
    plot = Button('plot')
    normalize = Bool(True)
    robust = Bool(True)
    cloud = Bool(False)
    se = Bool(False)
    n_samps = Int(100)
    dist_bin = Float(0.5)
    estimator = Enum('ergodic', ('ergodic', 'spot'))

    # animated variogram
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 20, 50, 100])
    plot_anim = Button('Animate variogram')

    # spatiotemporal kernel
    plot_acov = Button('Acov map')
    plot_stcov = Button('STCov map')
    plot_iso_stcov = Button('Isom. STCov map')
    n_lags = Int(10)
    norm_kernel = Bool(False)

    # tool button
    vis_tool = Button('Covar map')

    # stvar button
    stsemivar_tool = Button('Launch')
    st_len = Int(25)
    st_lag = Int(10)

    def _vis_tool_fired(self):
        _, y = self.curve_manager.interactive_curve.current_data()
        y *= 1e6
        if self.normalize:
            cov = np.corrcoef(y)
        else:
            cov = np.cov(y)
        chan_map = self.chan_map
        tool = ArrayVarianceTool(cov, chan_map)
        view = tool.default_traits_view()
        view.kind = 'live'
        tool.edit_traits(view=view)
        return tool

    def _stsemivar_tool_fired(self):
        _, y = self.curve_manager.interactive_curve.current_data()
        y *= 1e6
        chan_map = self.chan_map
        bin_size = None if (self.dist_bin < 0) else self.dist_bin
        tool = STSemivar.from_array_and_lag(
            y, self.st_len, self.st_lag, chan_map,
            normed=self.normalize,
            xbin=bin_size
        )
        view = tool.default_traits_view()
        view.kind = 'live'
        tool.edit_traits(view=view)
        return tool

    def _plot_acov_fired(self):
        t, y = self.curve_manager.interactive_curve.current_data(full_xdata=False)
        y *= 1e6
        t_stamp = t.mean()
        t = (t - t[0]) * 1e3
        dt = t[1] - t[0]
        N = int(self.n_lags / dt)
        Ct = autocov(y, all_lags=False)[:, :min(len(t), N)]
        if self.norm_kernel:
            Ct /= Ct[:, 0][:, None]
        chan_map = self.chan_map
        frames = chan_map.embed(Ct.T, axis=1)
        fig = Figure(figsize=(6, 10))
        ax1 = subplot2grid(fig, (2, 1), (0, 0))
        ax2 = subplot2grid(fig, (2, 1), (1, 0))

        # top panel is acov map at lag t
        mx = np.abs(Ct).max()
        _, cb = chan_map.image(Ct[:, 0], cmap='bwr', clim=(-mx, mx), ax=ax1)
        ax1.set_title('Map of  ~ t={0:.2f} sec'.format(t_stamp))

        # bottom panel left: acov as function of lag t
        ax2.plot(t[:len(frames)], Ct.T, lw=.25, color='k')
        ax2.set_xlabel('Lags (ms)')
        if self.norm_kernel:
            ax1.set_title('Map of autocorr.  ~ t={0:.2f} sec'.format(t_stamp))
            cb.set_label('Autocorrelation')
            ax2.set_ylabel('Autocorrelation')
        else:
            ax1.set_title('Map of autocov.  ~ t={0:.2f} sec'.format(t_stamp))
            cb.set_label('Autocovariance')
            ax2.set_ylabel('Autocovariance')
        sns.despine(ax=ax2)
        splot = MultiframeSavesFigure(fig, frames, mode_name='lag (ms)',
                                      frame_index=t[:len(frames)])
        splot.edit_traits()
        fig.tight_layout()
        fig.canvas.draw_idle()
        if not hasattr(self, '_kernel_plots'):
            self._kernel_plots = [splot]
        else:
            self._kernel_plots.append(splot)

    def _plot_iso_stcov_fired(self):
        t, y = self.curve_manager.interactive_curve.current_data(full_xdata=False)
        y *= 1e6
        t = (t - t[0]) * 1e3
        dt = t[1] - t[0]
        # shoot for minimum of 1 ms per frame
        skip = 1
        while dt * skip <= 1:
            skip += 1
        N = int(self.n_lags / (dt * skip))
        # print self.n_lags, dt, N, skip
        y = y - y.mean(axis=1, keepdims=1)
        N = min(len(t) - 1, N)
        Ct = np.empty((N, len(y), len(y)), 'd')
        T = len(t)
        for n in tqdm(range(0, skip * N, skip),
                      desc='Computing S-T functions', leave=True):
            Ct_ = np.einsum('ik,jk->ij', y[:, :T - n], y[:, n:])
            Ct[n // skip] = Ct_ / (T - n)

        chan_map = self.chan_map
        x_pts, y_pts = cxx_to_pairs(Ct, chan_map)
        # covar will be matrix with shape T, X
        covar = list()
        for y_ in y_pts:
            xb, yb = sv.binned_variance(x_pts, y_)
            covar.append(list(map(np.mean, yb)))

        covar = np.array(covar)
        if self.norm_kernel:
            covar /= covar[0, 0]
        # go ahead and plot T, X <--> ord., abs.
        # ... so covar.ravel() will count distance first, time second
        Xmesh = np.tile(xb[None, :], (N, 1))
        # T can use repeat
        Tmesh = np.repeat(np.arange(0, N) * (dt * skip), len(xb)).reshape(covar.shape)
        # Xmesh[0], Tmesh[0], Xmesh.shape, Tmesh.shape, covar.shape
        fig, ax = subplots(figsize=(6, 7))
        mn = covar.min()
        mx = covar.max()
        if mn * mx > 0:
            # same side of zero
            vmin = mn
            vmax = mx
            cmap = 'viridis'
        else:
            # spans zero
            c_lim = max(abs(mn), abs(mx))
            vmin = -c_lim
            vmax = c_lim
            cmap = 'bwr'

        im = ax.pcolormesh(
            Xmesh, Tmesh, covar,
            cmap=cmap, shading='flat', vmin=vmin, vmax=vmax
        )
        ax.set_ylabel('Time lag (ms)')
        ax.set_xlabel('Space lag (mm)')
        fig.colorbar(im, ax=ax)

        splot = SavesFigure.live_fig(fig)

    def _plot_stcov_fired(self):
        t, y = self.curve_manager.interactive_curve.current_data()
        y *= 1e6
        t_stamp = t[0].mean()
        t = (t - t[0]) * 1e3
        dt = t[1] - t[0]
        # shoot for minimum of 1 ms per frame
        skip = 1
        while dt * skip <= 1:
            skip += 1
        N = int(self.n_lags / (dt * skip))
        # print self.n_lags, dt, N, skip
        y = y - y.mean(axis=1, keepdims=1)
        N = min(len(t) - 1, N)
        Ct = np.empty((N, len(y), len(y)), 'd')
        T = len(t)
        for n in tqdm(range(0, skip * N, skip),
                      desc='Computing S-T functions', leave=True):
            Ct_ = np.einsum('ik,jk->ij', y[:, :T - n], y[:, n:])
            Ct[n // skip] = Ct_ / (T - n)

        if self.norm_kernel:
            nrm = np.sqrt(Ct[0].diagonal())
            Ct /= np.outer(nrm, nrm)

        chan_map = self.chan_map
        stcov = list()
        for Ct_ in tqdm(Ct, desc='Centering S-T kernels'):
            stcov.append(spatial_autocovariance(Ct_, chan_map, mean=True))
        stcov = np.array([np.nanmean(s_, axis=0) for s_ in stcov])
        y, x = stcov.shape[-2:]
        midx = int(x / 2)
        xx = (np.arange(x) - midx) * chan_map.pitch
        midy = int(y / 2)
        yy = (np.arange(y) - midy) * chan_map.pitch
        for n in range(N):
            stcov[n, midy, midx] = np.mean(Ct[n].diagonal())

        #fig, ax = subplots(figsize=(6, 5))
        fig = Figure(figsize=(6, 8))
        ax = subplot2grid(fig, (2, 2), (0, 0), colspan=2)
        mx = np.nanmax(np.abs(stcov))
        im = ax.imshow(
            stcov[0], cmap='bwr', clim=(-mx, mx),
            extent=[xx[0], xx[-1], yy[0], yy[-1]]
        )
        ax.set_xlabel('Distance X (mm)')
        ax.set_ylabel('Distance Y (mm)')
        ax.set_title('Spatiotemporal kernel ~ t={0:.2f} sec'.format(t_stamp))
        cb = fig.colorbar(im)
        if self.norm_kernel:
            cb.set_label('Autocorrelation map')
        else:
            cb.set_label('Autocovariance map')

        # Marginal over y
        ax2 = subplot2grid(fig, (2, 2), (1, 0))
        st_xt = np.nanmean(stcov, axis=1)
        mx = np.nanmax(np.abs(st_xt))
        im = ax2.imshow(
            st_xt, extent=[xx[0], xx[-1], t[N * skip], 0],
            clim=(-mx, mx), cmap='bwr', origin='upper'
        )
        ax2.axis('auto')
        fig.colorbar(im, ax=ax2)
        ax2.set_xlabel('Mean X-transect (mm)')
        ax2.set_ylabel('Lag (ms)')

        # Marginal over x
        ax3 = subplot2grid(fig, (2, 2), (1, 1))
        st_yt = np.nanmean(stcov, axis=2)
        mx = np.nanmax(np.abs(st_yt))
        im = ax3.imshow(
            st_yt, extent=[yy[0], yy[-1], t[N * skip], 0],
            clim=(-mx, mx), cmap='bwr', origin='upper'
        )
        ax3.axis('auto')
        fig.colorbar(im, ax=ax3)
        ax3.set_xlabel('Mean Y-transect (mm)')
        ax3.set_ylabel('Lag (ms)')

        splot = MultiframeSavesFigure(fig, stcov, mode_name='lag (ms)',
                                      frame_index=t[0:N * skip:skip])
        splot.edit_traits()
        fig.tight_layout()
        fig.canvas.draw_idle()
        if not hasattr(self, '_kernel_plots'):
            self._kernel_plots = [splot]
        else:
            self._kernel_plots.append(splot)

    def compute_variogram(self, array, chan_map, **kwargs):
        """
        Returns variogram points from the field defined by an array and channel map.
        The number of parameters returned depends on whether a variogram cloud is requested.
        If the cloud is not requested, then binned aggregates (x, y, se, Nd) are returned (that is: distance bins,
        mean semivariance at each distance, standard error of mean, and number of points in the original sample).
        Else, those binned aggregates are returned pre-pended by the non-aggregated (xcloud, ycloud) points.

        :param array:
        :param chan_map:
        :param kwargs:
        :return:
        """
        robust = kwargs.pop('robust', self.robust)
        n_samps = kwargs.pop('n_samps', self.n_samps)
        normed = kwargs.pop('normalize', self.normalize)
        cloud = kwargs.pop('cloud', self.cloud)
        if normed:
            array = array / np.std(array, axis=1, keepdims=1)

        pts = np.linspace(0, array.shape[1] - 1, n_samps).astype('i')
        if isinstance(chan_map, ChannelMap):
            combs = chan_map.site_combinations
        else:
            combs = chan_map

        # I think turn off binning if a cloud is requested
        bin_size = None if (self.dist_bin < 0 or cloud) else self.dist_bin
        # Turn off se if cloud is True
        se = not cloud
        counts = not cloud
        if self.estimator == 'ergodic':
            vg_method = partial(sv.fast_semivariogram, xbin=bin_size, trimmed=True, se=se, cloud=cloud, counts=counts)
        else:
            vg_method = partial(sv.semivariogram, robust=robust, xbin=bin_size, trimmed=True, se=se, cloud=cloud,
                                counts=counts)
        if cloud:
            bin_size = None if self.dist_bin < 0 else self.dist_bin
            xc, yc = vg_method(array[:, pts], combs)
            xb, yb = sv.binned_variance(xc, yc, binsize=bin_size)
            Nd = [len(y) for y in yb if len(y) > 1]
            # this will bring back mean and sem by default
            xb, yb, se = sv.binned_variance_aggregate(xb, yb)
            return xc, yc, xb, yb, se, Nd
        x, y, Nd, se = vg_method(array[:, pts], combs)
        return x, y, se, Nd

    def _plot_anim_fired(self):
        if not hasattr(self, '_animating'):
            self._animating = False
        if self._animating:
            self._f_anim._stop()
            self._animating = False
            return

        t, y = self.curve_manager.interactive_curve.current_data()
        y *= 1e6
        if self.normalize:
            y = y / np.std(y, axis=1, keepdims=1)
            #y = y / np.std(y, axis=0, keepdims=1)

        cm = self.chan_map

        t0 = time.time()
        r = self.compute_variogram(y[:, 0][:, None], cm, n_samps=1, normalize=False, cloud=False)
        t_call = time.time() - t0
        scaled_dt = self.anim_time_scale * (t[1] - t[0])

        f_skip = 1
        t_pause = scaled_dt - t_call / float(f_skip)
        while t_pause < 0:
            f_skip += 1
            t_pause = scaled_dt - t_call / float(f_skip)
            print('skipping', f_skip, 'samples and pausing', t_pause, 'sec')

        # this figure lives outside of figure "management"
        fig, axes = subplots()
        c = FigureCanvas(fig)

        x, svg, svg_se, Nd = r
        # also compute how long it takes to draw the errorbars
        line = axes.plot(x, svg, marker='s', ms=6, ls='--')[0]
        t0 = time.time()
        #ec = axes.errorbar(x, svg, yerr=svg_se, fmt='none', ecolor='gray')
        t_call = t_call + (time.time() - t0)
        t_pause = scaled_dt - t_call / float(f_skip)
        while t_pause < 0:
            f_skip += 1
            t_pause = scaled_dt - t_call / float(f_skip)
            print('skipping', f_skip, 'samples and pausing', t_pause, 'sec')
        # average together frames at the skip rate?
        N = y.shape[1]
        blks = N // f_skip
        y = y[:, :blks * f_skip].reshape(-1, blks, f_skip).mean(-1)
        # print y.mean()

        def _step(n):
            c.draw()
            print('frame', n)
            x, svg, svg_se, Nd = self.compute_variogram(
                y[:, n:n + 1], cm, n_samps=1, normalize=False
            )
            # ec.remove()
            line.set_data(x, svg)
            #ec = axes.errorbar(x, svg, yerr=svg_se, fmt='none', ecolor='gray')
            self.parent._qtwindow.vline.setPos(t[n * f_skip])
            if n == blks - 1:
                self._animating = False
            return (line,)

        c.show()

        max_var = 5.0 if self.normalize else y.var(0).max()
        axes.set_ylim(0, 1.05 * max_var)
        if not self.normalize:
            axes.axhline(max_var, color='k', ls='--', lw=2)

        #axes.autoscale(axis='both', enable=True)
        sns.despine(ax=axes)
        self._f_anim = FuncAnimation(
            fig, _step, blks, interval=scaled_dt * 1e3, blit=True
        )
        self._f_anim.repeat = False
        self._animating = True
        self._f_anim._start()

    def _plot_fired(self):
        t, y = self.curve_manager.interactive_curve.current_data()
        y *= 1e6
        r = self.compute_variogram(y, self.chan_map)
        if self.cloud:
            xc, yc, x, svg, svg_se, Nd = r
        else:
            x, svg, svg_se, Nd = r
        fig, ax = self._get_fig()
        label = 't ~ {0:.1f}'.format(t.mean())
        clr = self._colors[self._axplots[ax]]
        if self.cloud:
            ax.scatter(xc, yc, s=4, alpha=0.25, c=clr)
        ax.plot(
            x, svg, marker='s', ms=6, ls='--',
            color=clr, label=label
        )
        ax.errorbar(
            x, svg, svg_se, fmt='none',
            ecolor=clr
        )
        # if self.normalize:
        #     sill = 1.0
        #     ax.axhline(1.0, color=clr, ls='--')
        # else:
        #     sill = np.median(y.var(1))
        #     ax.axhline(sill, color=clr, ls='--')
        #ax.set_ylim(0, 1.05 * sill)
        ax.set_xlim(xmin=0)
        ax.autoscale(axis='y', enable=True)
        ax.set_ylim(ymin=0)
        ax.legend()
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Semivariance')
        sns.despine(ax=ax)
        fig.tight_layout()
        try:
            fig.canvas.draw_idle()
        except BaseException:
            pass

    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(
                    Label('Plot in new figure'),
                    UItem('new_figure'),
                    UItem('plot'),
                    UItem('vis_tool'),
                ),
                VGroup(
                    Item('estimator', label='Estimator'),
                    Item('cloud', label='Cloud'),
                    Item('se', label='Std Err'),
                    Item('n_samps', label='# of samples'),
                    Item('dist_bin', label='Distance bin (mm)'),
                    Item('normalize', label='Normalized'),
                    Item('robust', label='Robust'),
                    columns=2, label='Estimation params'
                ),
                VGroup(
                    VGroup(
                        Item('anim_time_scale', label='Time stretch'),
                        UItem('plot_anim'),
                        label='Animate variogram'
                    ),
                    VGroup(
                        HGroup(
                            Item('st_len', label='Short-time len'),
                            Item('st_lag', label='Short-time lag'),
                            UItem('stsemivar_tool')
                        ),
                        label='ST semivar tool'
                    )
                ),
                VGroup(
                    Item('n_lags',
                         editor=RangeEditor(low=0, high=1000, mode='slider'),
                         label='Time lags (ms)'),
                    HGroup(
                        UItem('plot_acov'),
                        UItem('plot_stcov'),
                        UItem('plot_iso_stcov')
                    ),
                    Item('norm_kernel', label='Normalized kernel'),
                    label='Spatiotemporal kernel'
                )
            )
        )
        return v
