from __future__ import division

import os, sys

import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
mpl.rcParams['interactive'] = True
mpl.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import \
     FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import numpy as np
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Instance, Button, HasTraits, Float, Bool, \
     Enum, on_trait_change, Str, List, Range, Property, Any, \
     property_depends_on, Int, File
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Group, ListEditor, VSplit, HSplit, Label, EnumEditor, \
     RangeEditor, FileEditor

from pyface.timer.api import Timer
import time

from tqdm import tqdm

from nitime.utils import autocov

from ecogana.anacode.anatool.gui_tools import SavesFigure, ArrayMap
from ecogana.anacode.signal_testing import centered_pair_maps
from ecogana.anacode.plot_util import filled_interval, subplots, \
     subplot2grid
from ecoglib.vis.traitsui_bridge import MPLFigureEditor, PingPongStartup
from ecoglib.estimation.jackknife import Jackknife
from ecoglib.numutil import nextpow2
from ecoglib.util import channel_combinations, ChannelMap
import ecoglib.estimation.multitaper as msp
import ecoglib.estimation.spatial_variance as sv
from sandbox.split_methods import multi_taper_psd

import ecogana.anacode.seaborn_lite as sns

class VisModule(HasTraits):
    name = Str('__dummy___')
    parent = Instance('VisWrapper')

class PlotsInterval(VisModule):
    new_figure = Bool(False)
    _figures = List([])

    def _get_fig(self, **kwargs):
        """Accepts subplots kwargs"""
        fig = getattr(self, 'fig', None)
        if fig and not self.new_figure:
            return fig, self.axes, fig.canvas.draw_idle
        self.fig, self.axes = subplots(**kwargs)
        splot = SavesFigure.live_fig(self.fig)
        self._figures.append(splot)
        self._plot_count = 0
        self._colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        return self.fig, self.axes, self.fig.canvas.draw_idle

class IntervalTraces(PlotsInterval):
    name = 'Plot Window'
    plot = Button('Plot')
    new_figure = Bool(True)

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        x = x[0]
        dy = self.parent.y_spacing
        f, ax, update = self._get_fig()
        clr = self._colors[self._plot_count]
        ax.plot(x, y.T + np.arange(len(y)) * dy, lw=0.5, color=clr)
        sns.despine(ax=ax)
        ax.set_ylabel('Amplitude (mV)')
        ax.set_xlabel('Time (s)')
        f.tight_layout()
        self._plot_count += 1
        try:
            update()
        except:
            pass

    def default_traits_view(self):
        v = View(
            Group(
                UItem('plot'),
                label='Plot current window'
                )
            )
        return v


class MultiframeSavesFigure(SavesFigure):
    _mx = Int(10)
    _mn = Int(0)
    mode = Int(0) #Range(low='_mn', high='_mx')
    mode_name = 'Mode'
    mode_value = Property(Float, depends_on='mode')

    def __init__(self, fig, frames, frame_index=(), **traits):
        super(MultiframeSavesFigure, self).__init__(fig, **traits)
        self.frames = frames
        self._mx = len(frames)-1
        if not len(frame_index):
            frame_index = range(len(frames))
        self.frame_index = frame_index

    def _get_mode_value(self):
        return np.round(self.frame_index[self.mode], decimals=2)
        
    @on_trait_change('mode')
    def change_frame(self):
        # has to assume fig.axes[0].images[0] has an image!
        im = self.fig.axes[0].images[0]
        im.set_array(self.frames[self.mode])
        self.fig.canvas.draw_idle()
    
    def default_traits_view(self):
        v = super(MultiframeSavesFigure, self).default_traits_view()
        vsplit = v.content.content[0]

        panel = HGroup(
            Item('mode', label='Scroll frames',
                 editor=RangeEditor(low=self._mn, high=self._mx)),
            UItem('mode', visible_when='_mx > 101',
                 editor=RangeEditor(
                     low=self._mn, high=self._mx, mode='slider'
                     )),
            Item('mode_value', style='readonly',
                 label='Current frame: {0}'.format(self.mode_name))
            )
        vsplit.content.insert(1, panel)
        return v

    @staticmethod
    def named_toggle(name):
        return MultiframeSavesFigure(mode_name=name)

class ArrayVarianceTool(HasTraits):
    """Stand-alone UI to image covariance seeded at chosen sites"""
    array_plot = Instance(ArrayMap)
    selected_site = Int(-1)
    min_max_norm = Bool(True)
    cmap = Enum('gray', ('gray', 'gray_r', 'jet', 'bwr', 'viridis'))
    save_file = File(os.getcwd())
    save_all = Button('Save PDF')

    def __init__(self, cov, chan_map, **traits):
        self.cov = cov
        self._cv_mn = cov.min()
        self._cv_mx = cov.max()
        array_plot = ArrayMap(chan_map)
        HasTraits.__init__(self, array_plot=array_plot)
        self.sync_trait('selected_site', self.array_plot, mutual=True)

    @on_trait_change('selected_site, cmap, min_max_norm')
    def _image(self):
        kw = dict(cmap=self.cmap)
        if self.min_max_norm:
            kw['clim'] = (self._cv_mn, self._cv_mx)
        else:
            if self._cv_mn < 0:
                # full neg-pos range
                mx = max( -self._cv_mn, self._cv_mx )
                kw['clim'] = (-mx, mx)
            else:
                kw['clim'] = (0, self._cv_mx)
        site = self.selected_site
        if site >= 0:
            self.array_plot.update_map(self.cov[site], **kw)
            self.array_plot.ax.set_title('Seeded covariance map')
        else:
            self.array_plot.update_map(self.cov.mean(1), **kw)
            self.array_plot.ax.set_title('Mean covariance weight')

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
        print tool.array_plot
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
        ax.set_xlabel('Distance Y (mm)')
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
        fig, ax, update = self._get_fig()
        label = 't ~ {0:.1f}'.format(t.mean())
        clr = self._colors[self._plot_count]
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
        self._plot_count += 1
        sns.despine(ax=ax)
        fig.tight_layout()
        try:
            update()
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
    
class IntervalSpectrum(PlotsInterval):
    name = 'Power Spectrum'
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    pow2 = Bool(True)
    avg_spec = Bool(True)
    sem = Bool(True)
    adaptive = Bool(True)
    plot = Button('Plot')

    def __default_mtm_kwargs(self, n):
        kw = dict()
        kw['NW'] = self.NW
        kw['adaptive'] = self.adaptive
        kw['Fs'] = self.parent.x_scale ** -1.0
        kw['NFFT'] = nextpow2(n) if self.pow2 else n
        kw['jackknife'] = False
        return kw
    
    def spectrum(self, array, **mtm_kw):
        kw = self.__default_mtm_kwargs(array.shape[-1])
        kw.update(mtm_kw)
        fx, pxx, _ = multi_taper_psd(array, **kw)
        return fx, pxx

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        fx, pxx = self.spectrum(y)

        fig, ax, update = self._get_fig()
                    
        label = 't ~ {0:.1f}'.format(x.mean())
        
        if self.avg_spec:
            jn = Jackknife(np.log(pxx), axis=0)
            mn, se = jn.estimate(np.mean, se=True)
            if not self.sem:
                # get jackknife stdev
                se = jn.estimate(np.std)
            ## l_pxx = np.log(pxx)
            ## mn = l_pxx.mean(0)
            ## se = l_pxx.std(0) / np.sqrt(len(l_pxx))
            se = np.exp( mn - se ), np.exp( mn + se )
            filled_interval(
                ax.semilogy, fx, np.exp(mn), se,
                color=self._colors[self._plot_count], label=label, ax=ax
                )
            ax.legend()
        else:
            lns = ax.semilogy(
                fx, pxx.T, lw=.25, color=self._colors[self._plot_count]
                )
            self._plot_count += 1
            ax.legend(lns[:1], (label,))
        ax.set_ylabel('Power Spectral Density (mV / Hz^2)')
        ax.set_xlabel('Frequency (Hz)')
        sns.despine(ax=ax)
        fig.tight_layout()
        self._plot_count += 1
        if update is not None:
            update()

    def default_traits_view(self):
        v = View(
            VGroup(
                HGroup(
                    Item('NW', label='NW'),
                    Item('pow2', label='Use radix-2 FFT'),
                    Item('adaptive', label='Adaptive MTM'),
                    ),
                HGroup(
                    Item('avg_spec', label='Plot avg'),
                    Item('sem', label='Use S.E.M.'),
                    Item('new_figure', label='Plot in new figure')
                    ),
                UItem('plot'),
                label='Spectrum plotting'
                )
            )
        return v

    
class IntervalSpectrogram(PlotsInterval):
    name = 'Spectrogram'

    channel = Str('all')
    _chan_list = Property(depends_on='parent.chan_map')
    # estimations details
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    _bandwidth = Property( Float )
    lag = Float(25) # ms
    strip = Float(100) # ms
    detrend = Bool(True)
    adaptive = Bool(True)
    over_samp = Range(0, 16, mode = 'spinner', value=4)
    high_res = Bool(True)

    baseline = Float(1.0) # sec
    normalize = Enum('Z score', ('None', 'Z score', 'divide', 'subtract'))
    plot = Button('Plot')
    freq_hi = Float
    freq_lo = Float(0)
    new_figure = Bool(True)

    colormap='Spectral_r'

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.freq_hi = round( (self.parent.x_scale ** -1.0) / 2.0 )
    
    def _get__chan_list(self):
        ii, jj = self.parent.chan_map.to_mat()
        clist = ['All']
        for i, j in zip(ii, jj):
            clist.append( '{0}, {1}'.format(i,j) )
        return sorted(clist)

    @property_depends_on('NW')
    def _get__bandwidth(self):
        #t1, t2 = self.parent._qtwindow.current_frame()
        T = self.strip * 1e-3
        # TW = NW --> W = NW / T
        return 2.0 * self.NW / T 
    
    def __default_mtm_kwargs(self, strip):
        kw = dict()
        kw['NW'] = self.NW
        kw['adaptive'] = self.adaptive
        Fs = self.parent.x_scale ** -1.0
        kw['Fs'] = Fs
        kw['jackknife'] = False
        kw['detrend'] = 'linear' if self.detrend else ''
        lag = int( Fs * self.lag / 1000. )
        if strip is None:
            strip = int( Fs * self.strip / 1000. )
        kw['NFFT'] = nextpow2(strip)
        kw['pl'] = 1.0 - float(lag) / strip
        if self.high_res:
            kw['samp_factor'] = self.over_samp
            kw['low_bias'] = 0.95
        return kw, strip

    def _normalize(self, tx, ptf, mode, tc):
        # ptf should be ([n_chan], n_freq, n_time)
        if not mode:
            mode = self.normalize
        if not tc:
            tc = self.baseline

        m = tx < tc
        if ptf.ndim < 3:
            ptf = ptf.reshape( (1,) + ptf.shape )
        nf = ptf.shape[1]
        p_bl = ptf[..., m].transpose(0, 2, 1).copy().reshape(-1, nf)

        # get mean of baseline for each freq.
        jn = Jackknife(p_bl, axis=0)
        mn = jn.estimate(np.mean)
        assert len(mn) == nf, '?'
        if mode.lower() == 'divide':
            ptf = ptf.mean(0) / mn[:, None]
        elif mode.lower() == 'subtract':
            ptf = np.exp(ptf.mean(0)) - np.exp(mn[:,None])
        elif mode.lower() == 'z score':
            stdev = jn.estimate(np.std)
            ptf = (ptf.mean(0) - mn[:,None]) / stdev[:,None]
        return ptf
        
    
    def highres_spectrogram(self, array, strip=None, **mtm_kw):
        kw, strip = self.__default_mtm_kwargs(strip)
        kw.update(**mtm_kw)
        tx, fx, ptf = msp.mtm_spectrogram(array, strip, **kw)
        n_cut = 6
        tx = tx[n_cut:-n_cut]
        ptf = ptf[..., n_cut:-n_cut].copy()
        return tx, fx, ptf

    def spectrogram(self, array, strip=None, **mtm_kw):
        kw, strip = self.__default_mtm_kwargs(strip)
        kw.update(**mtm_kw)
        tx, fx, ptf = msp.mtm_spectrogram_basic(array, strip, **kw)
        return tx, fx, ptf

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        x = x[0]
        if self.channel.lower() != 'all':
            i, j = map(int, self.channel.split(','))
            y = y[self.parent.chan_map.lookup(i, j)]

        if self.high_res:
            tx, fx, ptf = self.highres_spectrogram(y)
        else:
            tx, fx, ptf = self.spectrogram(y)

            
        if self.normalize.lower() != 'none':
            ptf = self._normalize(tx, ptf, self.normalize, self.baseline)
        else:
            ptf = np.log(ptf)
            if ptf.ndim > 2:
                ptf = ptf.mean(0)
        # cut out non-display frequencies and advance timebase to match window
        m = (fx >= self.freq_lo) & (fx <= self.freq_hi)
        ptf = ptf[m]
        fx = fx[m]
        tx += x[0]
        fig, ax, update = self._get_fig()
        im = ax.imshow(
            ptf, extent=[tx[0], tx[-1], fx[0], fx[-1]],
            cmap=self.colormap
            )
        ax.axis('auto')
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        cbar = fig.colorbar(im, ax=ax, shrink=0.5)
        cbar.locator = ticker.MaxNLocator(nbins=3)
        if self.normalize.lower() == 'divide':
            title = 'Normalized ratio'
        elif self.normalize.lower() == 'subtract':
            title = 'Baseline subtracted'
        elif self.normalize.lower() == 'z score':
            title = 'Z score'
        else:
            title = 'Log-power'
        cbar.set_label(title)
        fig.tight_layout()
        if update is not None:
            update()
        
    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    Label('Channel to plot'),
                    UItem('channel',
                          editor=EnumEditor(name='object._chan_list')),
                    UItem('plot'),
                    Label('Freq. Range'),
                    HGroup(
                        Item('freq_lo', label='low', width=3),
                        Item('freq_hi', label='high', width=3),
                        )
                    ),
                VGroup(
                    Item('high_res', label='High-res SG'),
                    Item('normalize', label='Normalize'),
                    Item('baseline', label='Baseline length (sec)'),
                    label='Spectrogram setup'
                    ),
                VGroup(
                    Item('NW'),
                    Item('_bandwidth', label='BW (Hz)',
                         style='readonly', width=4),
                    Item('strip', label='SG strip len (ms)'),
                    Item('lag', label='SG lag (ms)'),
                    Item('detrend', label='Detrend window'),
                    Item('adaptive', label='Adaptive MTM'),
                    label='Estimation details', columns=2
                    ),
                VGroup(
                    Item('over_samp', label='Oversamp high-res SG'),
                    enabled_when='high_res'
                    )
                )
            )
        return v
    
class YRange(VisModule):
    name = 'Trace Offsets'
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self.parent._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        v = View(
            Group(
                UItem('y_spacing'),
                label='Trace spacing (mv)'
                )
            )
        return v

class AnimateInterval(VisModule):
    name = 'Animate window'
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 20, 50, 100])

    def __step_frame(self):
        n = self.__n
        x = self.__x
        y = self.__y
        if n >= self.__n_frames:
            self._atimer.Stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self.parent._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            self._atimer.Stop()
        PySide.QtCore.QCoreApplication.instance().processEvents()
        # calculate the difference between the desired interval
        # and the time it just took to draw (per frame)
        t_pause = scaled_dt - (time.time() - t0) / self.__f_skip
        if t_pause < 0:
            self.__f_skip += 1
            print 'took', -t_pause, 'sec too long, increasing skip rate', self.__f_skip
            sys.stdout.flush()
        else:
            print t_pause
            self._atimer.setInterval(t_pause * 1000.0)
        self.__n += self.__f_skip

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer') and self._atimer.IsRunning():
            self._atimer.Stop()
            return
        
        x, self.__y = self.parent._qtwindow.current_data()
        self.__f_skip = 1
        self.__x = x[0]
        dt = self.__x[1] - self.__x[0]
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        print self.anim_time_scale * dt * 1000
        print self.__n_frames
        self._atimer = Timer( self.anim_time_scale * dt * 1000,
                              self.__step_frame )

    def default_traits_view(self):
        v = View(
            VGroup(
                Item('anim_time_scale', label='Divide real time'),
                Item('anim_frame'),
                label='Animate Frames'
                )
            )
        return v
    
def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])
    chan_map = Any
    
    modules = List( [IntervalTraces,
                     AnimateInterval,
                     IntervalSpectrum,
                     IntervalSpectrogram,
                     SpatialVariance] )

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)
        self._make_modules()

    def _make_modules(self):
        mods = self.modules[:]
        self.modules = []
        for m in mods:
            self.modules.append( m(parent=self) )

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        ht = 1000
        v = View(
            VSplit(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow.win),
                      resizable=True,
                      #height=(ht-150)),
                      height=0.85),
                HSplit(
                    Group(Label('Trace spacing (mv)'),
                          UItem('y_spacing', resizable=True)),
                                
                    UItem('modules', style='custom', resizable=True,
                          editor=ListEditor(use_notebook=True,
                                            deletable=False, 
                                            dock_style='tab',
                                            page_name='.name'),
                                            
                          height=-150),
                      )
                ),
            width=1200,
            height=ht,
            resizable=True,
            title='Quick Scanner'
        )
        return v
    
