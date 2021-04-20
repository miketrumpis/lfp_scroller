import numpy as np
from traits.api import Instance, Float, Enum, Any, List, on_trait_change, Event, Str, Property, Button
from traitsui.api import View, UItem, VSplit, CustomEditor, HSplit, Group, VGroup, HGroup, Label, ListEditor, \
    EnumEditor
from collections import OrderedDict
from pyqtgraph.Qt import QtGui
# from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
# from .pyqtgraph_extensions import get_colormap_lut
from pyqtgraph.colormap import listMaps, get as get_colormap

from .helpers import PersistentWindow
from .h5scroller import FastScroller
from .curve_collections import PlotCurveCollection
from .modules import *


def setup_qwidget_control(parent, editor, qwidget):
    return qwidget


class VisWrapper(PersistentWindow):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    _y_spacing_enum = Enum(200, [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 'entry'])
    _y_spacing_entry = Float(0.0)
    auto_space = Button('Auto y-space')
    colormap = Enum('coolwarm', listMaps(source='matplotlib'))
    y_spacing = Property
    vis_rate = Float
    chan_map = Any
    region = Event
    clear_selected_channels = Button('Clear selected')
    _plot_overlays = List(Str)
    active_channels_set = Str('all')
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
        self._plot_overlays = ['all', 'selected']
        self.overlay_lookup = dict(all=qtwindow.curve_collection,
                                   selected=qtwindow.selected_curve_collection)
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

    def _auto_space_fired(self):
        y = self._qtwindow.curve_collection.current_data(visible=True)[1]
        dy = np.median(y.ptp(axis=1)) * 1e6 * 0.25
        if round(dy) >= 1:
            dy = round(dy)
        self._y_spacing_enum = 'entry'
        self._y_spacing_entry = dy

    def add_new_curves(self, curves: PlotCurveCollection, label: str):
        self._qtwindow.p1.addItem(curves)
        self._plot_overlays.append(label)
        self.overlay_lookup[label] = curves

    def remove_curves(self, label: str):
        curves = self.overlay_lookup.pop(label, None)
        if curves is None:
            print('There was no overlay with label', label)
            return
        self._qtwindow.p1.removeItem(curves)
        self._plot_overlays.remove(label)

    def get_interactive_data(self, full_xdata=False):
        """
        Return the x-/y-visible data for the currently selected curve set.

        Returns
        -------
        x,y data: ndarrays
            The x_visible and y_visible arrays
        channel_map: ChannelMap
            The electrode map for the returned y-data

        """

        curves = self.overlay_lookup[self.active_channels_set]
        if self.active_channels_set != 'all':
            channel_map = curves.map_visible(self.chan_map)
        else:
            channel_map = self.chan_map
        x, y = curves.current_data(visible=True, full_xdata=full_xdata)
        return x, y, channel_map

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
                                        UItem('_y_spacing_entry'),
                                        UItem('auto_space')),
                                 VGroup(UItem('clear_selected_channels'),
                                        Label('Interactive curves'),
                                        UItem('active_channels_set',
                                              editor=EnumEditor(name='_plot_overlays')))),
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

