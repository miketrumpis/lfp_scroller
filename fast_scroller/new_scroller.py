from traits.api import Instance, Float, Enum, Any, List, on_trait_change, Event, Str, Property
from traitsui.api import View, UItem, VSplit, CustomEditor, HSplit, Group, Label, ListEditor

from pyqtgraph.Qt import QtGui

from .helpers import PersistentWindow
from .modules import *


def setup_qwidget_control(parent, editor, qwidget):
    return qwidget


class VisWrapper(PersistentWindow):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    _y_spacing_enum = Enum(200, [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 'entry'])
    _y_spacing_entry = Float(0.0)
    y_spacing = Property
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
        # Filter this keyword argument: this class can be called with "y_spacing" specified, even
        # though the "y_spacing" property on this class is read-only.
        dy = traits.pop('y_spacing', None)
        traits['_y_spacing_enum'] = dy
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
            module = m(parent=self, chan_map=self.chan_map, curve_collection=self._qtwindow.curve_collection)
            self.modules.append(module)

    def _get_y_spacing(self):
        if self._y_spacing_enum == 'entry':
            return float(self._y_spacing_entry)
        else:
            return float(self._y_spacing_enum)

    @on_trait_change('_y_spacing_enum')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing * 1e-6)

    @on_trait_change('_y_spacing_entry')
    def _change_spacing_from_entry(self):
        # only change data if we're in entry mode
        if self.y_spacing == self._y_spacing_entry:
            self._change_spacing()

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
                          UItem('_y_spacing_enum', resizable=True),
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

