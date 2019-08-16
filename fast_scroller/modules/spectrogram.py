import numpy as np
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

from traits.api import Button, Bool, Enum, Property, Float, \
     Range, HasTraits, cached_property
from traitsui.api import View, Group, VGroup, HGroup, Item, UItem, \
     Label, EnumEditor
     

from ecogdata.numutil import nextpow2
from ecoglib.estimation.jackknife import Jackknife
import ecoglib.estimation.multitaper as msp

from .base import PlotsInterval

__all__ = ['IntervalSpectrogram']

class IntervalSpectrogram(PlotsInterval):
    name = 'Spectrogram'

    ## channel = Str('all')
    ## _chan_list = Property(depends_on='parent.chan_map')
    # estimations details
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    lag = Float(25) # ms
    strip = Float(100) # ms
    detrend = Bool(True)
    adaptive = Bool(True)
    over_samp = Range(0, 16, mode = 'spinner', value=4)
    high_res = Bool(True)
    _bandwidth = Property( Float, depends_on='NW, strip' )

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
    
    #@property_depends_on('NW')
    @cached_property
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
        if not m.any():
            print('Baseline too short: using 2 bins')
            m[:2] = True
        if ptf.ndim < 3:
            ptf = ptf.reshape( (1,) + ptf.shape )
        nf = ptf.shape[1]
        p_bl = ptf[..., m].transpose(0, 2, 1).copy().reshape(-1, nf)

        # get mean of baseline for each freq.
        jn = Jackknife(p_bl, axis=0)
        mn = jn.estimate(np.mean)
        # smooth out mean across frequencies
        mn = gaussian_filter1d(mn, 0.05 * len(mn), mode='reflect')
        assert len(mn) == nf, '?'
        if mode.lower() == 'divide':
            ptf = ptf.mean(0) / mn[:, None]
        elif mode.lower() == 'subtract':
            ptf = np.exp(ptf.mean(0)) - np.exp(mn[:,None])
        elif mode.lower() == 'z score':
            stdev = jn.estimate(np.std)
            stdev = gaussian_filter1d(stdev, 0.05 * len(mn), mode='reflect')
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
        y *= 1e6
        x = x[0]
        if self.channel.lower() != 'all':
            i, j = list(map(float, self.channel.split(',')))
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
        fig, ax = self._get_fig()
        im = ax.imshow(
            ptf, extent=[tx[0], tx[-1], fx[0], fx[-1]],
            cmap=self.colormap, origin='lower'
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
        try:
            fig.canvas.draw_idle()
        except:
            pass
        
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
    
