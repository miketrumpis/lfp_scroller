import numpy as np
from ecoglib.vis.plot_util import AnchoredScaleBar
from ecoglib.vis import plotters
mpl = plotters.mpl
plt = plotters.plt
from ecogdata.channel_map import ChannelMap
from ecogdata.devices.units import nice_unit_text, best_scaling_step

from .base import FigureStack


class SimpleTiled:

    def __init__(self, xdata: np.ndarray, ydata: np.ndarray, channel_map: ChannelMap,
                 tile_size: tuple=(0.6, 0.4),
                 ax: mpl.axes.Axes=None,
                 y_units='uV',
                 x_units='s',
                 figstack: FigureStack=None):
        self.colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        self.channel_map = channel_map
        if ax is not None:
            self.ax = ax
            self.fig = ax.figure
        else:
            figsize = SimpleTiled.figsize(channel_map, tile_size=tile_size)
            if figstack is not None:
                self.fig, self.ax = figstack.new_figure(figsize=figsize)
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize)
        self._line_count = 0
        self.y_limits = ydata.min(), ydata.max()
        self.x_limits = xdata.min(), xdata.max()
        bar_info = dict()
        for bar, lim, units in zip(('x', 'y'), (self.x_limits, self.y_limits), (x_units, y_units)):
            gap = lim[1] - lim[0]
            bar_size, gain, bar_units = best_scaling_step(gap, units, allow_up=True)
            # Scale the bar_size to be in the data units
            bar_units = nice_unit_text(bar_units)
            bar_text = '{:d} {}'.format(int(bar_size), bar_units)
            bar_size = bar_size / gain
            bar_info[bar] = (bar_size, bar_text)
        self._bar_info = bar_info
        self.add_to_tiles(xdata, ydata, channel_map, initialize=True)
        # dirty hack to remember this tile controller
        self.ax.simple_tile = self

    @classmethod
    def figsize(cls, channel_map: ChannelMap, tile_size: tuple=(0.6, 0.4)):
        rows, cols = channel_map.geometry
        width = cols * tile_size[0]
        height = rows * tile_size[1]
        return (width, height)

    def add_to_tiles(self, xdata: np.ndarray, ydata: np.ndarray, channel_map: ChannelMap, initialize: bool=False):
        x_gap = 1.05 * (self.x_limits[1] - self.x_limits[0])
        y_gap = self.y_limits[1] - self.y_limits[0]

        # Store offsets as (x_off, y_off) and stack rows from top to bottom
        offsets = list()
        rows = channel_map.geometry[0]
        for i, j in zip(*channel_map.to_mat()):
            offsets.append((j * x_gap, (rows - i - 1) * y_gap))

        lines = list()
        # Assume xdata matches ydata, or can be tiled to do so
        if xdata.ndim == 1:
            xdata = np.tile(xdata, (len(ydata), 1))
        for n in range(len(ydata)):
            x0, y0 = offsets[n]
            lines.append(np.c_[xdata[n] + x0, ydata[n] + y0])
        lines = mpl.collections.LineCollection(lines, linewidths=0.5, colors=self.colors[self._line_count])
        self.ax.add_collection(lines)
        self._line_count += 1
        if initialize:
            self.ax.axis('off')
            # y_scale = y_gap / 3
            # y_scale = np.round(y_scale, decimals=-int(np.floor(np.log10(y_scale))))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.05)
            self.ax.autoscale_view(True, True, True)
            self.fig.canvas.draw()
            # Axes 0-1 coordinates
            data_loc = self.ax.transData.transform([xdata.min() - 0.1 * x_gap, ydata.min() - 0.05 * y_gap])
            scale_loc = self.ax.transAxes.inverted().transform(data_loc)

            y_size, y_text = self._bar_info['y']
            ybar = AnchoredScaleBar(scale_loc, self.ax, size=y_size, label=y_text, pad=0, borderpad=0,
                                    a_loc='lower right',
                                    vertical=True, linekw=dict(color='k', lw=1.5))
            x_size, x_text = self._bar_info['x']
            xbar = AnchoredScaleBar(scale_loc, self.ax, size=x_size, label=x_text, pad=0, borderpad=0,
                                    a_loc='upper left',
                                    vertical=False, linekw=dict(color='k', lw=1.5))
            self.ax.add_artist(ybar)
            self.ax.add_artist(xbar)







