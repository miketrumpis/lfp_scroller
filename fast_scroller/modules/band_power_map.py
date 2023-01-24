import numpy as np

from traits.api import Button, Bool, Float, Instance, Property, property_depends_on, Range, Enum
from traitsui.api import View, HGroup, Group, UItem, Item

from nitime.algorithms import dpss_windows

from .base import PlotsInterval, colormaps


class OneStopCache(dict):

    def __getitem__(self, k):
        return self.pop(k, dpss_windows(*k))

    def __setitem__(self, k, v):
        for key_ in set(self.keys()) - {k}:
            self.pop(key_)
        super(OneStopCache, self).__setitem__(k, v)


class BandPowerMap(PlotsInterval):

    name = 'Frequency Map'
    plot = Button('Plot')
    new_figure = Property(Bool)

    BW = Enum(1, (1, 2, 3, 4, 5))
    cmap = Enum(colormaps[0], colormaps)
    fc = Range('_f_lo', '_f_hi', value=2)
    sub_block_pct = Enum(10, (5, 10, 15, 20, 25))

    _f_lo = Property(Float, depends_on='BW')
    _f_hi = Property(Float, depends_on='BW')
    _dpss_cache = Instance(OneStopCache, ())

    @property_depends_on('BW')
    def _get__f_hi(self):
        dt = self.curve_manager.interactive_curve.dx
        return int(np.floor(0.5 * (dt ** -1) - self.BW))

    @property_depends_on('BW')
    def _get__f_lo(self):
        return int(self.BW)

    def _get_new_figure(self):
        return True

    def _plot_fired(self):
        x, y = self.curve_manager.interactive_curve.current_data(full_xdata=False)
        # convert to microvolts
        y *= 1e6
        t = 0.5 * (x[0] + x[-1])
        dt = x[1] - x[0]
        # N = int( self.sub_block_pct * 0.01 * y.shape[-1] )
        N = y.shape[-1]

        #K = 2 * N * self.BW * dt - 1
        NW = N * self.BW * dt
        K = 2 * NW - 1
        Kmax = min(100, K)

        # Kmax seems redundant, but .. why not?
        # dpss_key = (N, self.BW, Kmax)
        dpss_key = (N, NW, Kmax)
        dpss, eigs = self._dpss_cache[dpss_key]
        ei = np.exp(1j * 2 * np.pi * self.fc * dt * np.arange(N))
        dpss = dpss * ei
        # yp, dpss = moving_projection(
        # y, N, self.BW, dt**-1, f0=self.fc, Kmax=Kmax, baseband=True,
        ##     dpss=dpss, save_dpss=True
        # )
        self._dpss_cache[dpss_key] = (dpss, eigs)

        ## band_power = yp.std(-1)

        # get DPSS coefficients (n_chan x Kmax)
        c = y.dot(dpss.T)
        band_power = np.sqrt(np.sum(np.abs(c)**2, axis=1) / N)

        f, ax = self._get_fig(figsize=(5, 5))
        chan_map = self.chan_map
        f, cb = chan_map.image(band_power, ax=ax, cmap=self.cmap)
        cb.set_label('Bandpass RMS (uV)')
        ax.set_title(
            '{0:0.1f} \u00b1 {1} Hz bandpass ~ {2:0.1f} s'.format(self.fc, self.BW, t)
        )

    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    UItem('plot'),
                    Item('cmap', label='Colormap')
                ),
                Group(
                    Item('fc', label='Center frequency'),
                    Item('BW', label='Half-bandwidth')
                )
            )
        )
        return v
