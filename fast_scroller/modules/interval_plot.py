import numpy as np

from mpl_toolkits.axes_grid1 import AxesGrid

from traits.api import Button, Bool, Enum, Int
from traitsui.api import View, HGroup, VGroup, Group, Item, UItem, Label

import ecogana.anacode.seaborn_lite as sns

from .base import PlotsInterval, SimpleFigure, colormaps

def _hh_mm_ss(t):
    h = int(t // 3600)
    t -= h * 3600
    m = int(t // 60)
    t -= m * 60
    return '{h:02d}:{m:02d}:{s:02d}'.format(h=h, m=m, s=int(t))

class IntervalTraces(PlotsInterval):
    name = 'Plot Window'
    plot = Button('Plot')
    new_figure = Bool(True)
    label_channels = Bool(False)

    map_row = Enum(1, range(1, 11))
    map_col = Enum(1, range(1, 11))
    map_rms = Bool(True)
    cmaps = Enum('gray', colormaps)
    new_maps = Button('Create map grid')
    insert_map = Button('Map now')
    mm_per_pixel = Int(3)

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        y *= 1e6
        x = x[0]
        dy = self.parent.y_spacing
        f, ax = self._get_fig()
        clr = self._colors[self._axplots[ax]]
        y_levels = np.arange(len(y)) * dy
        ax.plot(x, y.T + y_levels, lw=0.5, color=clr)
        sns.despine(ax=ax)
        if not self.label_channels:
            ax.set_ylabel('Amplitude (mV)')
        else:
            ax.set_yticks( y_levels )
            ii, jj = self.parent._qtwindow.chan_map.to_mat()
            labels = map(str, zip(ii, jj))
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Time (s)')
        f.tight_layout()
        try:
            f.canvas.draw_idle()
        except:
            pass

    def _new_maps_fired(self):
        self.create_maps()

    def create_maps(self):
        g = self.parent._qtwindow.chan_map.geometry
        # this is an over-estimate of inches per pixel, maybe
        # find another rule
        inch_per_pixel = 0.039 * self.mm_per_pixel
        # size of array map: width, height
        img_size = g[1] * inch_per_pixel, g[0] * inch_per_pixel
        figsize = self.map_col * img_size[0], self.map_row * img_size[1]
        figwin = SimpleFigure(figsize=figsize)
        fig = figwin.figure

        text_size_inch = 11 / fig.dpi
        
        grid = AxesGrid(
            fig, 111, nrows_ncols=(self.map_row, self.map_col),
            axes_pad=1.5 * text_size_inch,
            cbar_mode='single', cbar_location='right',
            cbar_pad='2%', cbar_size='4%'
            )

        for ax in grid:
            ax.axis('off')
        self._grid = grid
        self._g_idx = 0
        self._cbar = None
        # hold onto this or it disappears
        figwin.show()
        self._figwin = figwin
            
    def _insert_map_fired(self):
        self.map_current_frame()

    def map_current_frame(self):
        if not hasattr(self, '_grid'):
            print 'create figure forced'
            self.create_maps()
        ax = self._grid[self._g_idx]
        self._g_idx = (self._g_idx + 1) % len(self._grid)
        chan_map = self.parent._qtwindow.chan_map

        x, y = self.parent._qtwindow.current_data()
        y *= 1e6
        x = x[0]
        if self.map_rms:
            img = chan_map.embed(y.std(1))
            label = '~ '+_hh_mm_ss( 0.5 * (x[0] + x[-1]))
        else:
            t = self.parent._qtwindow.vline.getPos()[0]
            print t
            if (x[0] <= t) and (t <= x[-1]):
                img = chan_map.embed(y[:, x.searchsorted(t)])
            else:
                t = x[0]
                img = chan_map.embed(y[:, 0])
            label = _hh_mm_ss(t)

        clim = self.parent._qtwindow.img.getLevels()
        clim = (1e6 * clim[0], 1e6 * clim[1])
        im = ax.imshow(img, origin='upper', clim=clim, cmap=self.cmaps)
        ax.set_title(label, fontsize=8)
        if self._cbar is None:
            self._cbar = self._grid.cbar_axes[0].colorbar(im)
        self._figwin.canvas.draw_idle()

    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    UItem('plot'),
                    Label('Label Channels'),
                    UItem('label_channels'),
                    label='Plot current window'
                    ),
                Group(
                    HGroup(
                        VGroup(
                            UItem('new_maps'),
                            UItem('insert_map'),
                            Item('map_rms', label='Map RMS?'),
                            ),
                        VGroup(
                            Label('Rows/Columns'),
                            HGroup(UItem('map_row'), UItem('map_col'))
                            ),
                        VGroup(
                            Label('Colormap'),
                            UItem('cmaps')
                            )
                        ),
                    label='Voltage heatmaps'
                    )
                )
            )
        return v
