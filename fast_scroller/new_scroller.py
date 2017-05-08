from __future__ import division

import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams['interactive'] = True
mpl.use('Qt4Agg')
from matplotlib.figure import Figure
import sys
import numpy as np
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Instance, Button, HasTraits, Float, Bool, \
     Enum, on_trait_change, Str, List, Range, Property, Any, \
     property_depends_on
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Group, ListEditor, VSplit, HSplit, Label, EnumEditor, VFlow

from pyface.timer.api import Timer
import time

from ecogana.anacode.anatool.gui_tools import SavesFigure
from ecogana.anacode.plot_util import filled_interval, subplots
from ecoglib.estimation.jackknife import Jackknife
from ecoglib.numutil import nextpow2
import ecoglib.estimation.multitaper as msp
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
        m = (fx >= self.freq_lo) & (fx <= self.freq_hi)
        ptf = ptf[m]
        fx = fx[m]
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
                     IntervalSpectrogram] )

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
    
