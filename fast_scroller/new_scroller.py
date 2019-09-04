from traits.api import Instance, Float, Enum, Any, List, on_trait_change, Event, Str
from traitsui.api import View, UItem, VSplit, CustomEditor, HSplit, Group, Label, ListEditor

from pyqtgraph.Qt import QtGui

from .helpers import PersistentWindow
from .modules import *

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(PersistentWindow):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    y_spacing = Enum(200, [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 'entry'])
    _y_spacing_entry = Float(0.0)
    chan_map = Any
    region = Event
    modules = List( [IntervalTraces,
                     AnimateInterval,
                     IntervalSpectrum,
                     IntervalSpectrogram,
                     SpatialVariance] )
    recording = Str

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        super(VisWrapper, self).__init__(**traits)
        # connect the region-changed signal to signal a traits callback
        qtwindow.region.sigRegionChanged.connect(self._region_did_change)
        self._make_modules()

    def _region_did_change(self):
        self.region = True

    def _make_modules(self):
        mods = self.modules[:]
        self.modules = []
        for m in mods:
            self.modules.append( m(parent=self) )

    @on_trait_change('y_spacing, _y_spacing_entry')
    def _change_spacing(self):
        if self.y_spacing == 'entry':
            y_spacing = self._y_spacing_entry
        else:
            y_spacing = self.y_spacing
        self._qtwindow.update_y_spacing(y_spacing * 1e-6)

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
                    Group(Label('Trace spacing (uV)'),
                          UItem('y_spacing', resizable=True),
                          Label('Enter spacing (uV)'),
                          UItem('_y_spacing_entry')),
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
            title=self.recording
        )
        return v

