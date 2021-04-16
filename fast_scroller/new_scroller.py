from traits.api import Instance, Float, Enum, Any, List, on_trait_change, Event, Str, Property, Button
from traitsui.api import View, UItem, VSplit, CustomEditor, HSplit, Group, VGroup, HGroup, Label, ListEditor
from collections import OrderedDict
from pyqtgraph.Qt import QtGui
# from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
# from .pyqtgraph_extensions import get_colormap_lut
from pyqtgraph.colormap import listMaps, get as get_colormap

from .helpers import PersistentWindow
from .h5scroller import FastScroller
from .modules import *


def setup_qwidget_control(parent, editor, qwidget):
    return qwidget


class VisWrapper(PersistentWindow):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    _y_spacing_enum = Enum(200, [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 'entry'])
    _y_spacing_entry = Float(0.0)
    colormap = Enum('coolwarm', listMaps(source='matplotlib'))
    y_spacing = Property
    vis_rate = Float
    chan_map = Any
    region = Event
    clear_selected_channels = Button('Clear selected')
    modules = List([IntervalTraces,
                    AnimateInterval,
                    IntervalSpectrum,
                    IntervalSpectrogram,
                    SpatialVariance])
    recording = Str

    def __init__(self, qtwindow: FastScroller, **traits):
        self._qtwindow = qtwindow
        # Filter this keyword argument: this class can be called with "y_spacing" specified, even
        # though the "y_spacing" property on this class is read-only.
        dy = traits.pop('y_spacing', 100)
        traits['_y_spacing_enum'] = dy
        traits['vis_rate'] = 1 / qtwindow.curve_collection.dx
        super(VisWrapper, self).__init__(**traits)
        self._change_colormap()
        qtwindow.region.sigRegionChanged.connect(self._region_did_change)
        qtwindow.curve_collection.vis_rate_changed.connect(self._follow_vis_rate)
        self._make_modules()

    def _region_did_change(self):
        self.region = True

    def _follow_vis_rate(self, rate):
        self.vis_rate = rate

    def _make_modules(self):
        mods = self.modules[:]
        self.modules_by_name = OrderedDict()
        for m in mods:
            module = m(parent=self, chan_map=self.chan_map, curve_collection=self._qtwindow.curve_collection,
                       selected_curve_collection=self._qtwindow.selected_curve_collection)
            self.modules_by_name[module.name] = module
        self.modules = list(self.modules_by_name.values())

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

    def _clear_selected_channels_fired(self):
        self._qtwindow.selected_curve_collection.set_selection(None)

    @on_trait_change('colormap')
    def _change_colormap(self):
        cmap = get_colormap(self.colormap, source='matplotlib')
        self._qtwindow.cb.setCmap(cmap)
        # self._qtwindow.img.setLookupTable(cmap)

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
                    Group(HGroup(VGroup(Label('Trace spacing (uV)'),
                                        UItem('_y_spacing_enum', resizable=True),
                                        Label('Enter spacing (uV)'),
                                        UItem('_y_spacing_entry')),
                                 UItem('clear_selected_channels')),
                          Label('Color map'),
                          UItem('colormap'),
                          HGroup(Label('Vis. sample rate'), UItem('vis_rate')),
                          ),
                    UItem('modules', style='custom', resizable=True,
                          editor=ListEditor(use_notebook=True,
                                            deletable=False, 
                                            dock_style='tab',
                                            page_name='.name'),
                                            
                          # height=-150
                          ),
                      )
                ),
            width=1200,
            height=ht,
            resizable=True,
            title=self.recording
        )
        return v

