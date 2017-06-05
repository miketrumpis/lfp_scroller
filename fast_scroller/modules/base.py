from collections import OrderedDict
import numpy as np
import matplotlib as mpl
from traits.api import HasTraits, Str, Instance, Bool, Int, Property, \
     Float, on_trait_change
from traitsui.api import Item, RangeEditor, UItem, HGroup

from ecogana.anacode.plot_util import subplots
from ecogana.anacode.anatool.gui_tools import SavesFigure

class MultiframeSavesFigure(SavesFigure):
    """
    Specialization of SavesFigure that has a plotting element
    with multiple frames that can be scanned-through.
    """
    
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

class FigureStack(OrderedDict):

    def new_figure(self, **kwargs):
        fig, axes = subplots(**kwargs)
        splot = SavesFigure.live_fig(fig)
        
        def remove_fig():
            print 'catch closing'
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
        k = self.keys()[-1]
        return self[k]

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
    channel = Str('all')
    _chan_list = Property(depends_on='parent.chan_map')

    def _get__chan_list(self):
        ii, jj = self.parent.chan_map.to_mat()
        clist = ['All']
        for i, j in zip(ii, jj):
            clist.append( '{0}, {1}'.format(i,j) )
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

