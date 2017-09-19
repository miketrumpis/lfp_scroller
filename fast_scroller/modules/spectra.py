import numpy as np

from traits.api import Button, Bool, Enum, Property, Float, property_depends_on
from traitsui.api import View, VGroup, HGroup, Item, UItem
     

from ecoglib.numutil import nextpow2
from ecoglib.estimation.jackknife import Jackknife
from sandbox.split_methods import multi_taper_psd

from ecogana.anacode.plot_util import filled_interval
import ecogana.anacode.seaborn_lite as sns

from .base import PlotsInterval

__all__ = ['IntervalSpectrum']

class IntervalSpectrum(PlotsInterval):
    name = 'Power Spectrum'
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    _bandwidth = Property( Float )
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

    @property_depends_on('NW')
    def _get__bandwidth(self):
        t1, t2 = self.parent._qtwindow.current_frame()
        T = t2 - t1
        # TW = NW --> W = NW / T
        return 2.0 * self.NW / T 
    
    def spectrum(self, array, **mtm_kw):
        kw = self.__default_mtm_kwargs(array.shape[-1])
        kw.update(mtm_kw)
        fx, pxx, _ = multi_taper_psd(array, **kw)
        return fx, pxx

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        fx, pxx = self.spectrum(y)

        fig, ax = self._get_fig()
                    
        label = 't ~ {0:.1f}'.format(x.mean())
        plot_count = self._axplots[ax]
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
                color=self._colors[plot_count], label=label, ax=ax
                )
            ax.legend()
        else:
            lns = ax.semilogy(
                fx, pxx.T, lw=.25, color=self._colors[plot_count]
                )
            ax.legend(lns[:1], (label,))
        ax.set_ylabel('Power Spectral Density (mV / Hz^2)')
        ax.set_xlabel('Frequency (Hz)')
        sns.despine(ax=ax)
        fig.tight_layout()
        try:
            fig.canvas.draw_idle()
        except:
            pass

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
                HGroup(
                    UItem('plot'),
                    Item('_bandwidth', label='BW (Hz)',
                         style='readonly', width=4),
                    ),
                label='Spectrum plotting'
                )
            )
        return v

    