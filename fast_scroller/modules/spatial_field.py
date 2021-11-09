import os
import time
from functools import partial

import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FuncAnimation

from traits.api import Instance, Button, HasTraits, Float, Bool, Enum, on_trait_change, Int, File
from traitsui.api import View, VGroup, HGroup, Item, UItem, VSplit, Label, FileEditor

from tqdm import tqdm

from ecogdata.channel_map import ChannelMap
from ecoglib.estimation.spatial_variance import semivariogram, fast_semivariogram, binned_variance, \
    binned_variance_aggregate
from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.vis.gui_tools import ArrayMap
from ecoglib.vis.plot_util import subplots
from ecoglib.vis import plotters

from .base import PlotsInterval, colormaps, FigureCanvas
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
            vg_method = partial(fast_semivariogram, xbin=bin_size, trimmed=True, se=se, cloud=cloud, counts=counts)
        else:
            vg_method = partial(semivariogram, robust=robust, xbin=bin_size, trimmed=True, se=se, cloud=cloud,
                                counts=counts)
        if cloud:
            bin_size = None if self.dist_bin < 0 else self.dist_bin
            xc, yc = vg_method(array[:, pts], combs)
            xb, yb = binned_variance(xc, yc, binsize=bin_size)
            # this will bring back mean and sem by default
            xb, yb, se, Nd = binned_variance_aggregate(xb, yb)
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
        plotters.sns.despine(ax=axes)
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
        plotters.sns.despine(ax=ax)
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
                    Item('anim_time_scale', label='Time stretch'),
                    UItem('plot_anim'),
                    label='Animate variogram'
                )
            )
        )
        return v
