from collections import Counter

import numpy as np
import scipy.optimize as so

from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from traits.api import Instance, Button, Bool, on_trait_change, Int
from traitsui.api import View, VGroup, HGroup, Item, UItem, HSplit, VSplit, RangeEditor

from tqdm import tqdm

from nitime.utils import autocov

from ecogdata.channel_map import ChannelMap
from ecogdata.filt.blocks import BlockedSignal
import ecoglib.estimation.spatial_variance as sv
from ecoglib.signal_testing import spatial_autocovariance
from ecoglib.estimation import cxx_to_pairs, matern_semivariogram, matern_spectrum, make_matern_label
from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.vis.gui_tools import SavesFigure
from ecoglib.vis.plot_util import subplots, subplot2grid
from ecoglib.vis import plotters

from .base import MultiframeSavesFigure
from . import SpatialVariance
from ..helpers import PersistentWindow, view_label_item


__all__ = ['STSemivar', 'ExtraSpatialVariance']


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
        plotters.sns.despine(ax=ax)
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

    @classmethod
    def from_array_and_lag(cls, x, n, lag, chan_map, normed=False, **kwargs):
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
        return cls(sx, block_time, st_svar, x, chan_map)

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
        # [::-1] is to transpose the (row, col) to (x, y) coordinates
        lines = [(i2[::-1], i1[::-1])
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


class ExtraSpatialVariance(SpatialVariance):
    name = 'Spatial Variance (Dev)'

    # spatiotemporal kernel
    plot_acov = Button('Acov map')
    plot_stcov = Button('STCov map')
    plot_iso_stcov = Button('Isom. STCov map')
    n_lags = Int(10)
    norm_kernel = Bool(False)

    # stvar button
    stsemivar_tool = Button('Launch')
    st_len = Int(25)
    st_lag = Int(10)

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
        plotters.sns.despine(ax=ax2)
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

    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(
                    view_label_item('dist_bin', 'Distance bin (mm)', vertical=True),
                    view_label_item('normalize', 'Normalize variance', vertical=True),
                    label='Semivariance options'
                ),
                VGroup(
                    Item('st_len', label='Short-time len'),
                    Item('st_lag', label='Short-time lag'),
                    UItem('stsemivar_tool'),
                    label='ST semivar tool'),
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
