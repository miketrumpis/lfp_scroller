import os
import time
import numpy as np
from matplotlib.backends.backend_qt4agg import \
     FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from traits.api import Instance, Button, HasTraits, Float, Bool, \
     Enum, on_trait_change, Int, File
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     VSplit, Label, RangeEditor, FileEditor

from tqdm import tqdm

from nitime.utils import autocov

from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.util import channel_combinations, ChannelMap
from ecoglib.estimation.jackknife import Jackknife
import ecoglib.estimation.spatial_variance as sv
from ecogana.anacode.signal_testing import centered_pair_maps
from ecogana.anacode.anatool.gui_tools import ArrayMap
import ecogana.anacode.seaborn_lite as sns
from ecogana.anacode.plot_util import subplots, subplot2grid


from .base import PlotsInterval, MultiframeSavesFigure, colormaps

__all__ = ['ArrayVarianceTool', 'SpatialVariance']

class ArrayVarianceTool(HasTraits):
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
        array_plot = ArrayMap(chan_map)
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
        # cycle through sites, update image, and use
        # PdfPages.savefig(self.array_plot.fig)
        chan_map = self.array_plot.chan_map
        ij = zip(*chan_map.to_mat())
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
    jackknife = Bool(False)
    n_samps = Int(100)
    pitch = Float(1.0)
    estimator = Enum('ergodic', ('ergodic', 'spot'))

    # animated variogram
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 20, 50, 100])
    plot_anim = Button('Animate variogram')

    # spatiotemporal kernel
    plot_acov = Button('Acov map')
    plot_stcov = Button('STCov map')
    n_lags = Int(10)
    norm_kernel = Bool(False)

    # tool button
    vis_tool = Button('Covar map')

    def _vis_tool_fired(self):
        _, y = self.parent._qtwindow.current_data()
        if self.normalize:
            cov = np.corrcoef(y)
        else:
            cov = np.cov(y)
        chan_map = self.parent._qtwindow.chan_map
        tool = ArrayVarianceTool(cov, chan_map)
        view = tool.default_traits_view()
        view.kind = 'live'
        tool.edit_traits(view=view)
        return tool

    def _plot_acov_fired(self):
        t, y = self.parent._qtwindow.current_data()
        t_stamp = t[0].mean()
        t = (t[0] - t[0,0]) * 1e3
        dt = t[1] - t[0]
        N = int(self.n_lags / dt)
        Ct = autocov(y, all_lags=False)[:, :min(len(t), N)]
        if self.norm_kernel:
            Ct /= Ct[:,0][:,None]
        chan_map = self.parent._qtwindow.chan_map
        frames = chan_map.embed(Ct.T, axis=1)
        fig = Figure(figsize=(6, 10))
        ax1 = subplot2grid(fig, (2, 1), (0, 0))
        ax2 = subplot2grid(fig, (2, 1), (1, 0))
        
        # top panel is acov map at lag t
        mx = np.abs(Ct).max()
        _, cb = chan_map.image(Ct[:,0], cmap='bwr', clim=(-mx, mx), ax=ax1)
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


    def _plot_stcov_fired(self):
        t, y = self.parent._qtwindow.current_data()
        t_stamp = t[0].mean()
        t = (t[0] - t[0,0]) * 1e3
        dt = t[1] - t[0]
        # shoot for minimum of 1 ms per frame
        skip = 1
        while dt*skip <= 1:
            skip += 1
        N = int(self.n_lags / (dt * skip))
        print self.n_lags, dt, N, skip
        y = y - y.mean(axis=1, keepdims=1)
        N = min( len(t)-1, N )
        Ct = np.empty( (N, len(y), len(y)), 'd' )
        T = len(t)
        for n in tqdm(xrange(0, skip*N, skip),
                      desc='Computing S-T functions', leave=True):
            Ct_ = np.einsum('ik,jk->ij', y[:, :T-n], y[:, n:])
            Ct[n//skip] = Ct_ / (T-n)

        if self.norm_kernel:
            nrm = np.sqrt( Ct[0].diagonal() )
            Ct /= np.outer(nrm, nrm)

        chan_map = self.parent._qtwindow.chan_map
        chan_combs = channel_combinations(chan_map, scale=self.pitch)

        idx1 = chan_combs.idx1; idx2 = chan_combs.idx2

        d_idx = np.triu_indices(len(y), k=1)
        stcov = list()
        for Ct_ in tqdm(Ct, desc='Centering S-T kernels'):
            stcov.append( centered_pair_maps(Ct_[d_idx], idx1, idx2) )
        stcov = np.array([np.nanmean(s_, axis=0) for s_ in stcov])
        y, x = stcov.shape[-2:]
        midx = int( x/2 ); xx = (np.arange(x) - midx) * self.pitch
        midy = int( y/2 ); yy = (np.arange(y) - midy) * self.pitch
        for n in xrange(N):
            stcov[n, midy, midx] = np.mean(Ct[n].diagonal())

        #fig, ax = subplots(figsize=(6, 5))
        fig = Figure(figsize=(6, 8))
        ax = subplot2grid(fig, (2, 2), (0, 0), colspan=2)
        mx = np.nanmax(np.abs(stcov))
        print mx
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
        ax2 = subplot2grid(fig, (2, 2), (1, 0) )
        st_xt = np.nanmean(stcov, axis=1)
        mx = np.nanmax(np.abs(st_xt))
        print mx
        im = ax2.imshow(
            st_xt, extent=[xx[0], xx[-1], t[N*skip], 0],
            clim=(-mx, mx), cmap='bwr', origin='upper'
            )
        ax2.axis('auto')
        fig.colorbar(im, ax=ax2)
        ax2.set_xlabel('Mean X-transect (mm)')
        ax2.set_ylabel('Lag (ms)')

        # Marginal over x
        ax3 = subplot2grid(fig, (2, 2), (1, 1) )
        st_yt = np.nanmean(stcov, axis=2)
        mx = np.nanmax(np.abs(st_yt))
        print mx
        im = ax3.imshow(
            st_yt, extent=[yy[0], yy[-1], t[N*skip], 0],
            clim=(-mx, mx), cmap='bwr', origin='upper'
            )
        ax3.axis('auto')
        fig.colorbar(im, ax=ax3)
        ax3.set_xlabel('Mean Y-transect (mm)')
        ax3.set_ylabel('Lag (ms)')
                
        splot = MultiframeSavesFigure(fig, stcov, mode_name='lag (ms)',
                                      frame_index=t[0:N*skip:skip])
        splot.edit_traits()
        fig.tight_layout()
        fig.canvas.draw_idle()
        if not hasattr(self, '_kernel_plots'):
            self._kernel_plots = [splot]
        else:
            self._kernel_plots.append(splot)


    def compute_variogram(self, array, chan_map, **kwargs):
        scale = kwargs.pop('scale', self.pitch)
        robust = kwargs.pop('robust', self.robust)
        n_samps = kwargs.pop('n_samps', self.n_samps)
        normed = kwargs.pop('normalize', self.normalize)
        jackknife = kwargs.pop('jackknife', self.jackknife)
        if normed:
            array = array / np.std(array, axis=1, keepdims=1)
        
        pts = np.linspace(0, array.shape[1]-1, n_samps).astype('i')
        if isinstance(chan_map, ChannelMap):
            combs = channel_combinations(chan_map, scale=scale)
        else:
            combs = chan_map
        x = np.unique(combs.dist)
        svg = np.empty( (len(pts), len(x)) )
        for n in xrange(len(pts)):
            _, svg[n] = sv.semivariogram(array[:,pts[n]], combs, robust=robust, trimmed=False)
        if jackknife:
            jn = Jackknife(svg, axis=0)
            mn, se = jn.estimate(np.mean, se=True)
        else:
            mn = svg.mean(0)
            se = svg.std(0) / np.sqrt(len(svg))
        return x, mn, se

    def compute_ergodic_variogram(self, array, chan_map, **kwargs):
        scale = kwargs.pop('scale', self.pitch)
        n_samps = kwargs.pop('n_samps', self.n_samps)
        normed = kwargs.pop('normalize', self.normalize)
        if normed:
            array = array / np.std(array, axis=1, keepdims=1)
        pts = np.linspace(0, array.shape[1]-1, n_samps).astype('i')
        ## combs = channel_combinations(chan_map, scale=scale)
        ## x = np.unique(combs.dist)
        import ecogana.anacode.spatial_profiles as sp
        svg = sv.ergodic_semivariogram(array[:, pts])
        x, svg = sp.cxx_to_pairs(svg, chan_map, pitch=scale)
        xb, svg, se = sp.binned_obs_error(*sp.binned_obs(x, svg))
        return xb, svg, np.sqrt(se)

    def _plot_anim_fired(self):
        if not hasattr(self, '_animating'):
            self._animating = False
        if self._animating:
            self._f_anim._stop()
            self._animating = False
            return
        
        t, y = self.parent._qtwindow.current_data()
        t = t[0]
        if self.normalize:
            y = y / np.std(y, axis=1, keepdims=1)
            #y = y / np.std(y, axis=0, keepdims=1)

        if self.estimator == 'ergodic':
            fn = self.compute_ergodic_variogram
            cm = self.parent._qtwindow.chan_map
        else:
            fn = self.compute_variogram
            cm = self.parent._qtwindow.chan_map
            cm = channel_combinations(cm, scale=self.pitch)
            
        t0 = time.time()
        r = fn(y[:, 0][:,None], cm, n_samps=1, normalize=False, jackknife=False)
        t_call = time.time() - t0
        scaled_dt = self.anim_time_scale * (t[1] - t[0])
        
        f_skip = 1
        t_pause = scaled_dt - t_call / float(f_skip)
        while t_pause < 0:
            f_skip += 1
            t_pause = scaled_dt - t_call / float(f_skip)
            print 'skipping', f_skip, 'samples and pausing', t_pause, 'sec'

        # this figure lives outside of figure "management"
        fig, axes = subplots()
        c = FigureCanvas(fig)

        x, svg, svg_se = r
        # also compute how long it takes to draw the errorbars
        line = axes.plot(x, svg, marker='s', ms=6, ls='--')[0]
        t0 = time.time()
        #ec = axes.errorbar(x, svg, yerr=svg_se, fmt='none', ecolor='gray')
        t_call = t_call + (time.time() - t0)
        t_pause = scaled_dt - t_call / float(f_skip)
        while t_pause < 0:
            f_skip += 1
            t_pause = scaled_dt - t_call / float(f_skip)
            print 'skipping', f_skip, 'samples and pausing', t_pause, 'sec'
        
        # average together frames at the skip rate?
        N = y.shape[1]
        blks = N // f_skip
        y = y[:, :blks*f_skip].reshape(-1, blks, f_skip).mean(-1)
        print y.mean()
        def _step(n):
            c.draw()
            print 'frame', n
            x, svg, svg_se = fn(
                y[:, n:n+1], cm, n_samps=1, normalize=False, jackknife=False
                )
            #ec.remove()
            line.set_data(x, svg)
            #ec = axes.errorbar(x, svg, yerr=svg_se, fmt='none', ecolor='gray')
            self.parent._qtwindow.vline.setPos(t[n * f_skip])
            if n == blks-1:
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
            fig, _step, blks, interval=scaled_dt*1e3, blit=True
            )
        self._f_anim.repeat = False
        self._animating = True
        self._f_anim._start()

    def _plot_fired(self):
        t, y = self.parent._qtwindow.current_data()
        if self.estimator == 'ergodic':
            r = self.compute_ergodic_variogram(
                y, self.parent._qtwindow.chan_map
                )
        else:
            r = self.compute_variogram(y, self.parent._qtwindow.chan_map)
        x, svg, svg_se = r
        fig, ax = self._get_fig()
        label = 't ~ {0:.1f}'.format(t.mean())
        clr = self._colors[self._axplots[ax]]
        ax.plot(
            x, svg, marker='s', ms=6, ls='--',
            color=clr, label=label
            )
        ax.errorbar(
            x, svg, svg_se, fmt='none',
            ecolor=clr
            )
        if self.normalize:
            sill = 1.0
            ax.axhline(1.0, color=clr, ls='--')
        else:
            sill = np.median(y.var(1))
            ax.axhline(sill, color=clr, ls='--')
        #ax.set_ylim(0, 1.05 * sill)
        ax.autoscale(axis='y', enable=True)
        ax.set_ylim(ymin=0)
        ax.legend()
        ax.set_xlabel('Distance (mm)')
        ax.set_ylabel('Semivariance')
        sns.despine(ax=ax)
        fig.tight_layout()
        try:
            fig.canvas.draw_idle()
        except:
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
                    Item('jackknife', label='Jackknife'),
                    Item('n_samps', label='# of samples'),
                    Item('pitch', label='Electrode pitch'),
                    Item('normalize', label='Normalized'),
                    Item('robust', label='Robust'),
                    columns=2, label='Estimation params'
                    ),
                VGroup(
                    Item('anim_time_scale', label='Time stretch'),
                    UItem('plot_anim'),
                    label='Animate variogram'
                    ),
                VGroup(
                    Item('n_lags',
                         editor=RangeEditor(low=0, high=1000, mode='slider'),
                         label='Time lags (ms)'),
                    HGroup(UItem('plot_acov'), UItem('plot_stcov')),
                    Item('norm_kernel', label='Normalized kernel'),
                    label='Spatiotemporal kernel'
                    )
                )
            )
        return v
