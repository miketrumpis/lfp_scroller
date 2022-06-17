"""Module for helpful code (mostly traits-based mods)."""

import os
from time import time
from collections import OrderedDict
from traits.api import HasTraits, Str, Enum, Instance, on_trait_change, List, Property
from traitsui.api import View, VGroup, spring, HGroup, Item, Color, UItem, Label, ListEditor, Handler, EnumEditor
from pyface.timer.api import do_after
from ecogdata.parallel.mproc import parallel_context, timestamp

from .curve_collections import PlotCurveCollection


# A HasTraits "mixin" that keeps a reference to its own UI window so that
# QT doesn't destroy the window if it goes out of focus.
# Note: implementing as a direct subclass (rather than mixin) to avoid
# weird multiple inheritance metaclass conflicts.
# To do correctly: http://mcjeff.blogspot.com/2009/05/odd-python-errors.html

class _window_holder:

    def __init__(self, window):
        self._hold_ui = window

    def __call__(self):
        delattr(self, '_hold_ui')


class PersistentWindow(HasTraits):

    def configure_traits(self, **args):
        ui = super(PersistentWindow, self).configure_traits(**args)
        del_ui = _window_holder(ui)
        ui.control.destroyed.connect(del_ui)
        # TODO: this handle didn't help with the out-of-focus closing
        # self._ui_ctrl = ui.control
        return ui

    def edit_traits(self, **args):
        ui = super(PersistentWindow, self).edit_traits(**args)
        del_ui = _window_holder(ui)
        ui.control.destroyed.connect(del_ui)
        # TODO: this handle didn't help with the out-of-focus closing
        # self._ui_ctrl = ui.control
        return ui


# generic error message pop-up
class Error(HasTraits):
    error_msg = Str('')
    view = View(
        VGroup(
            spring,
            HGroup(Item('error_msg', style='readonly'), show_labels=False),
            spring
            ),
        buttons=['OK']
    )


class PlotItemSettings(HasTraits):
    curve_name = Str
    pen_color = Color
    pen_width = Enum(1, (0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8))
    shadowpen_color = Color
    shadowpen_width = Enum(1, (0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8))
    curves = Instance('fast_scroller.curve_collections.PlotCurveCollection')

    def __init__(self, curves, **traits):
        for pen_name in ('pen', 'shadowPen'):
            trait_name = pen_name.lower()
            pen = curves.opts[pen_name]
            if pen is None:
                traits[trait_name + '_width'] = 0
            else:
                color = pen.color()
                traits[trait_name + '_color'] = (color.red(), color.blue(), color.green(), color.alpha())
                traits[trait_name + '_width'] = pen.width()
        traits['curves'] = curves
        super().__init__(**traits)

    @on_trait_change('pen_color, pen_width')
    def set_pen(self):
        if self.curves is None:
            return
        if self.pen_width == 0:
            self.curves.setPen(None)
        else:
            self.curves.setPen(self.pen_color, width=self.pen_width)

    @on_trait_change('shadowpen_color, shadowpen_width')
    def set_shadowpen(self):
        if self.curves is None:
            return
        if self.shadowpen_width == 0:
            self.curves.setShadowPen(None)
        else:
            self.curves.setShadowPen(color=self.shadowpen_color, width=self.shadowpen_width)

    def default_traits_view(self):
        v = View(
            VGroup(
                HGroup(VGroup(Label('Width'), UItem('pen_width')),
                       VGroup(Label('Color'), UItem('pen_color', style='custom')),
                       label='Main Curve Settings'),
                HGroup(VGroup(Label('Width'), UItem('shadowpen_width')),
                       VGroup(Label('Color'), UItem('shadowpen_color', style='custom')),
                       label='Curve Shadow Settings'),
            )
        )
        return v


class CurveManager(HasTraits):
    plot = Instance('pyqtgraph.graphicsItems.PlotItem.PlotItem')
    # So... PlotItemSettings has both a label and a PlotCurveCollection.
    # It can be used to produce the label list and to retrieve the actual curves
    curve_settings = List(PlotItemSettings)
    # collections = List(PlotCurveCollection)
    _curve_names = Property(depends_on='curve_settings')
    # Curve that's displayed on the heatmap
    heatmap_name = Str
    heatmap_curve = Property(depends_on='_heatmap_name')
    # Curve that's available downstream
    interactive_name = Str
    interactive_curve = Property(depends_on='_interactive_name')
    # TODO: "private" (should not be set more than once)
    _source_curve_name = Str
    source_curve = Instance(PlotCurveCollection)

    def _get__curve_names(self):
        return [s.curve_name for s in self.curve_settings]

    def curves_by_name(self, label, settings=False):
        curves = [s for s in self.curve_settings if s.curve_name == label]
        if len(curves):
            return curves[0] if settings else curves[0].curves
        return None

    def _get_heatmap_curve(self):
        return self.curves_by_name(self.heatmap_name, settings=False)

    def _get_interactive_curve(self):
        return self.curves_by_name(self.interactive_name, settings=False)

    def add_new_curves(self, curves: PlotCurveCollection, label: str, set_source: bool=False):
        self.plot.addItem(curves)
        self.curve_settings.append(PlotItemSettings(curves, curve_name=label))
        if not self.heatmap_name:
            self.heatmap_name = label
        if not self.interactive_name:
            self.interactive_name = label
        if set_source and not self._source_curve_name:
            self._source_curve_name = label
            self.source_curve = curves

    def remove_curves(self, label: str):
        curve_settings = self.curves_by_name(label, settings=True)
        if curve_settings is None:
            print('There was no overlay with label', label)
            return
        self.plot.removeItem(curve_settings.curves)
        self.curve_settings.remove(curve_settings)
        if self.heatmap_name == label:
            self.heatmap_name = self._curve_names[0]
        if self.interactive_name == label:
            self.heatmap_name = self._curve_names[0]
        if self._source_curve_name == label:
            self._source_curve_name = ''

    def default_traits_view(self):
        v = View(
            VGroup(
                HGroup(Item('heatmap_name', label='Curves for heatmap',
                            editor=EnumEditor(name='_curve_names')),
                       Item('interactive_name', label='Curves for modules',
                            editor=EnumEditor(name='_curve_names'))),
                UItem('curve_settings', style='custom', resizable=True,
                      editor=ListEditor(use_notebook=True,
                                        deletable=False,
                                        dock_style='tab',
                                        page_name='.curve_name'))
            ),
            resizable=True
        )
        return v


class DebounceCallback:
    debounce_interval_ms = 200

    def __init__(self, callback):
        """
        A de-bounced dispatcher.

        Parameters
        ----------
        callback: Python callable
            The callable normally attached to signal.connect

        """
        self.callback = callback
        self._signal_times = list()

    @classmethod
    def connect(cls, signal, callback, interval=None):
        debounced = cls(callback)
        if interval is not None:
            debounced.debounce_interval_ms = interval
        signal.connect(debounced.debounced_callback)
        # return the callable that can be disconnected
        return debounced.debounced_callback

    def debounced_callback(self, *emitting):
        """
        This callback can be connected to a signal emitting object.
        When this signal debouncer is idle, the first signal will be dispatched and then follow-up signals
        will be ignored if they follow too closely. A final dispatch will be attempted at the end of the debounce
        interval.

        Parameters
        ----------
        emitting: QtObject
            The emitting object.

        """
        now = time()
        info = parallel_context.get_logger().info
        info('callback for {}, now is {}'.format(self.callback, self._signal_times))
        if not len(self._signal_times):
            # dispatch now and start a new timer
            self.apply_callback(*emitting)
            self._signal_times.append(now)
            do_after(self.debounce_interval_ms, self.apply_callback, *emitting)
        else:
            if (now - self._signal_times[-1]) < self.debounce_interval_ms / 1000:
                # put this time on the end of the stack
                self._signal_times.append(now)
            else:
                # if more than the debounce interval has elapsed, clear the stack
                # and call this callback again
                self._signal_times = list()
                self.debounced_callback(*emitting)

    def apply_callback(self, *emitting):
        """
        Apply the real callback

        Parameters
        ----------
        emitting: QtObject
            The emitting object

        """
        # if the signal time stack is only one deep, then that signal was already dispatched
        if len(self._signal_times) == 1:
            return
        info = parallel_context.get_logger().info
        info('applying callback {} {} {}'.format(self.callback, emitting, timestamp()))
        self.callback(*emitting)
        self._signal_times = list()


def validate_file_path(f):
    f = os.path.abspath(f)
    exists = os.path.exists(os.path.dirname(f))
    not_dir = not os.path.isdir(f)
    return exists and not_dir


def view_label_item(trait: str, label: str, vertical: bool=False, label_first: bool=True,
                    item_kwargs: dict=None, **group_kwargs):
    l_view = Label(label)
    if item_kwargs is None:
        item_kwargs = dict()
    t_view = UItem(trait, **item_kwargs)
    g_type = VGroup if vertical else HGroup
    args = (l_view, t_view) if label_first else (t_view, l_view)
    return g_type(*args, **group_kwargs)