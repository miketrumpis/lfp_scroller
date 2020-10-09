from collections import OrderedDict
import numpy as np

from pyqtgraph.Qt import QtGui

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import AxesGrid

from traits.api import HasTraits, Str, Instance, Bool, Int, Property, Float, on_trait_change, File, Button
from traitsui.api import Item, RangeEditor, UItem, HGroup, FileEditor

from ecoglib.vis.ani import write_anim
from ecoglib.vis.plot_util import subplots
from ecoglib.vis.gui_tools import SavesFigure

from ..helpers import Error, validate_file_path

# useful colormaps
colormaps = ['gray', 'jet', 'bwr', 'viridis', 'Blues', 'winter',
             'inferno', 'coolwarm', 'BrBG']
colormaps.extend([c + '_r' for c in colormaps])
colormaps = sorted(colormaps)

# A no-frills QT window for MPL Figure (modified from SO:
# https://stackoverflow.com/questions/12459811/how-to-embed-matplotlib-in-pyqt-for-dummies


class SimpleFigure(QtGui.QDialog):
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
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)


class MultiframeSavesFigure(SavesFigure):
    """
    Specialization of SavesFigure that has a plotting element
    with multiple frames that can be scanned-through.
    """

    _mx = Int(10)
    _mn = Int(0)
    mode = Int(0)  # Range(low='_mn', high='_mx')
    mode_name = 'Mode'
    mode_value = Property(Float, depends_on='mode')
    _has_ffmpeg = Bool
    video_file = File
    video_fps = Int(10)
    make_video = Button('Make video')

    def __init__(self, fig, frames, frame_index=(), **traits):
        import matplotlib.animation as anim
        traits['_has_ffmpeg'] = 'ffmpeg' in anim.writers.list()
        super(MultiframeSavesFigure, self).__init__(fig, **traits)
        self.frames = frames
        self._mx = len(frames)-1
        if not len(frame_index):
            frame_index = list(range(len(frames)))
        self.frame_index = frame_index

    def _get_mode_value(self):
        return np.round(self.frame_index[self.mode], decimals=2)

    @on_trait_change('mode')
    def change_frame(self):
        # has to assume fig.axes[0].images[0] has an image!
        im = self.fig.axes[0].images[0]
        im.set_array(self.frames[self.mode])
        self.fig.canvas.draw_idle()

    def _make_video_fired(self):
        if not validate_file_path(self.video_file):
            ev = Error(
                error_msg='Invalid video file:\n{0}'.format(self.video_file)
            )
            ev.edit_traits()
            return
        mode = self.mode

        def step_fn(n):
            self.mode = n
            return (self.fig.axes[0].images[0],)
        write_anim(
            self.video_file, self.fig, step_fn, self._mx,
            quicktime=True
        )
        self.mode = mode

    def default_traits_view(self):
        v = super(MultiframeSavesFigure, self).default_traits_view()
        vsplit = v.content.content[0]

        if self._has_ffmpeg:
            vpanel = HGroup(
                Item('video_file', editor=FileEditor(dialog_style='save')),
                Item('video_fps', label='FPS',
                     editor=RangeEditor(low=1, high=100, mode='spinner')),
                UItem('make_video')
            )
            vsplit.content.insert(1, vpanel)
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


class AxesGridStack(FigureStack):

    def _create_figure(self, **kwargs):
        # creates a figure and then an AxesGrid in the 111 position

        # other than figsize, no other Figure kwargs allowed
        figsize = kwargs.pop('figsize', None)
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
    curve_collection = Instance('fast_scroller.h5scroller.CurveCollection')
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
