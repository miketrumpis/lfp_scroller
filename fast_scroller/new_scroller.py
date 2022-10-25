import numpy as np
from traits.api import Instance, Float, Enum, Any, List, on_trait_change, Event, Str, Property, Button
from traitsui.api import View, UItem, VSplit, CustomEditor, HSplit, Group, VGroup, HGroup, Label, ListEditor, \
    EnumEditor
from collections import OrderedDict
from pyqtgraph.Qt import QtGui
from pyqtgraph.colormap import listMaps, get as get_colormap

from .helpers import PersistentWindow, CurveManager
from .h5scroller import FastScroller
from .modules import *


def setup_qwidget_control(parent, editor):
    widget = editor.object.graph
    return widget


class VisWrapper(PersistentWindow):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    _y_spacing_enum = Enum(200, [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 'entry'])
    _y_spacing_entry = Float(0.0)
    auto_space = Button('Auto y-space')
    auto_clim = Button('Auto colorbar')
    colormap = Enum('coolwarm', listMaps(source='matplotlib'))
    y_spacing = Property
    vis_rate = Float
    chan_map = Any
    region = Event
    clear_selected_channels = Button('Clear selected')
    curve_editor_popup = Button('Curve settings')
    # _plot_overlays = List(Str)
    # active_channels_set = Str('all')
    curve_manager = Instance(CurveManager)
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
        traits['vis_rate'] = 1 / qtwindow.curve_manager.source_curve.dx
        traits['curve_manager'] = qtwindow.curve_manager
        traits['graph'] = qtwindow.win
        super(VisWrapper, self).__init__(**traits)
        self._change_colormap()
        # self._plot_overlays = ['all', 'selected']
        # self.overlay_lookup = dict(all=qtwindow.curve_collection,
        #                            selected=qtwindow.selected_curve_collection)
        qtwindow.region.sigRegionChanged.connect(self._region_did_change)
        qtwindow.curve_manager.source_curve.vis_rate_changed.connect(self._follow_vis_rate)
        self._make_modules()

    # This is here only to announce to traits callbacks that the region has changed
    def _region_did_change(self):
        self.region = True

    def _follow_vis_rate(self, rate):
        self.vis_rate = rate

    def _make_modules(self):
        mods = self.modules[:]
        self.modules_by_name = OrderedDict()
        for m in mods:
            module = m(parent=self, chan_map=self.chan_map, curve_manager=self.curve_manager)
            self.modules_by_name[module.name] = module
        self.modules = list(self.modules_by_name.values())

    def _get_y_spacing(self):
        if self._y_spacing_enum == 'entry':
            return float(self._y_spacing_entry)
        else:
            return float(self._y_spacing_enum)

    def _auto_space_fired(self):
        y = self.curve_manager.source_curve.current_data(visible=True)[1]
        dy = np.median(y.ptp(axis=1)) * 1e6 * 0.25
        if round(dy) >= 1:
            dy = round(dy)
        self._y_spacing_enum = 'entry'
        self._y_spacing_entry = dy

    def _auto_clim_fired(self):
        image = self._qtwindow.img.image
        mn, mx = np.nanpercentile(image, [2, 98])
        # If bipolar, keep +/- limits same magnitude
        if mn * mx > 0:
            limits = mn, mx
        else:
            mx = max(np.abs(mn), np.abs(mx))
            limits = -mx, mx
        self._qtwindow.cb.setLevels(values=limits)

    @on_trait_change('_y_spacing_enum')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing * 1e-6)

    @on_trait_change('_y_spacing_entry')
    def _change_spacing_from_entry(self):
        # only change data if we're in entry mode
        if self.y_spacing == self._y_spacing_entry:
            self._change_spacing()

    def _clear_selected_channels_fired(self):
        self.curve_manager.curves_by_name('selected').set_selection(None)

    def _curve_editor_popup_fired(self):
        # curve_tool = CurveColorSettings(self.overlay_lookup)
        curve_tool = self.curve_manager
        v = curve_tool.default_traits_view()
        v.kind = 'live'
        curve_tool.edit_traits(view=v)

    @on_trait_change('colormap')
    def _change_colormap(self):
        cmap = get_colormap(self.colormap, source='matplotlib')
        self._qtwindow.cb.setColorMap(cmap)
        # self._qtwindow.img.setLookupTable(cmap)

    def default_traits_view(self):
        ht = 1000
        v = View(
            VSplit(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control),
                      resizable=True,
                      #height=(ht-150)),
                      height=0.85),
                HSplit(
                    Group(HGroup(VGroup(Label('Trace spacing (uV)'),
                                        UItem('_y_spacing_enum', resizable=True),
                                        Label('Enter spacing (uV)'),
                                        UItem('_y_spacing_entry'),
                                        UItem('auto_space')),
                                 VGroup(Label('Heatmap curves'),
                                        UItem('object.curve_manager.heatmap_name',
                                              editor=EnumEditor(name='object.curve_manager._curve_names')),
                                        Label('Color map'),
                                        UItem('colormap'),
                                        UItem('auto_clim'))),
                          HGroup(VGroup(UItem('clear_selected_channels'),
                                        UItem('curve_editor_popup')),
                                 VGroup(Label('Interactive curves'),
                                        UItem('object.curve_manager.interactive_name',
                                              editor=EnumEditor(name='object.curve_manager._curve_names')))),
                          HGroup(Label('Vis. sample rate'), UItem('vis_rate', style='readonly'))),
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

