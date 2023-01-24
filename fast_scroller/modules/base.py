from collections import OrderedDict
from pyqtgraph.Qt import QtWidgets

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from .. import QT
from ..uicore import SavesFigure

if QT != 'qt4':
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from mpl_toolkits.axes_grid1 import AxesGrid
from traits.api import HasTraits, Str, Instance, Bool, Property
from ecoglib.vis.plot_util import subplots


# useful colormaps
colormaps = ['gray', 'jet', 'bwr', 'viridis', 'Blues', 'winter',
             'inferno', 'coolwarm', 'BrBG']
colormaps.extend([c + '_r' for c in colormaps])
colormaps = sorted(colormaps)

# A no-frills QT window for MPL Figure (modified from SO:
# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies


class SimpleFigure(QtWidgets.QWidget):
    def __init__(self, parent=None, **fig_kwargs):
        super(SimpleFigure, self).__init__(parent)

        # a figure instance to plot on
        self.figure = Figure(**fig_kwargs)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)


class FigureStack(OrderedDict):

    def _create_figure(self, **kwargs):
        fig, axes = subplots(**kwargs)
        return fig, axes

    def new_figure(self, **kwargs):
        fig, axes = self._create_figure(**kwargs)
        splot = SavesFigure.live_fig(fig)

        def remove_fig():
            try:
                self.pop(splot)
            except ValueError:
                return

        fig.canvas.destroyed.connect(remove_fig)
        self[splot] = (fig, axes)
        return fig, axes

    def current_figure(self):
        if not len(self):
            return
        k = list(self.keys())[-1]
        return self[k]


class GridSpecStack(FigureStack):

    def _create_figure(self, figsize=None, nrows=1, ncols=1, **kwargs):
        # creates a figure and then a GridSpec to make subplots
        fig = Figure(figsize=figsize)
        gs = GridSpec(nrows, ncols, figure=fig, **kwargs)
        return fig, gs


class AxesGridStack(FigureStack):

    def _create_figure(self, figsize=None, **kwargs):
        # creates a figure and then an AxesGrid in the 111 position
        fig = Figure(figsize=figsize)
        grid = AxesGrid(fig, 111, **kwargs)
        self.grid = grid
        self.next_grid = {grid: 0}
        return fig, grid

    def current_frame(self):
        _, grid = self.current_figure()
        g_idx = self.next_grid[grid]
        frame = grid[g_idx]
        # roll-around counting
        g_idx = (g_idx + 1) % len(grid)
        self.next_grid[grid] = g_idx
        return frame


class PlotCounter(dict):

    def __getitem__(self, a):
        if a in self:
            n = super(PlotCounter, self).__getitem__(a)
            self[a] = n + 1
            return n
        self[a] = 0
        return self[a]


class VisModule(HasTraits):
    name = Str('__dummy___')
    parent = Instance('fast_scroller.new_scroller.VisWrapper')
    chan_map = Instance('ecogdata.channel_map.ChannelMap')
    # curve_collection = Instance('fast_scroller.curve_collections.PlotCurveCollection')
    # selected_curve_collection = Instance('fast_scroller.curve_collections.LabeledCurveCollection')
    curve_manager = Instance('fast_scroller.helpers.CurveManager')
    channel = Str('all')
    _chan_list = Property(depends_on='parent.chan_map')

    def _get__chan_list(self):
        ii, jj = self.chan_map.to_mat()
        clist = ['All']
        for i, j in zip(ii, jj):
            clist.append('{0}, {1}'.format(i, j))
        return sorted(clist)


class PlotsInterval(VisModule):
    new_figure = Bool(False)
    _figstack = Instance(FigureStack, args=())
    _axplots = Instance(PlotCounter, args=())

    def _get_fig(self, **kwargs):
        if len(self._figstack) and not self.new_figure:
            fig, axes = self._figstack.current_figure()
            return fig, axes
        self._colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        return self._figstack.new_figure(**kwargs)

    @property
    def page_limits(self):
        qt_graph = self.parent._qtwindow
        minX, maxX = qt_graph.region.getRegion()
        return minX, maxX
