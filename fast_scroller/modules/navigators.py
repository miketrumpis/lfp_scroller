from functools import partial
import numpy as np
import pyqtgraph as pg

from traits.api import Button, Str, on_trait_change, Bool, Float, HasTraits, Instance, List, Enum
from traitsui.api import View, HGroup, Group, UItem, Handler, ListEditor

from .base import VisModule
from ..h5data import h5stat
from ..filtering import AutoPanel

__all__ = ['NavigatorManager']


class NavigatorBuilder(AutoPanel):
    orientation = 'horizontal'
    visible = Bool(False, info_text='Visible')
    color = Str('r', info_text='Trace color')
    width = Float(0.5, info_text='Trace width')

    def __init__(self, array, x_scale, y_scale, rowmask, **traits):
        super(NavigatorBuilder, self).__init__(**traits)
        self._array = array
        self._x_scale = x_scale
        self._y_scale = y_scale
        self._rowmask = rowmask
        self._curves = ()

    @on_trait_change('color, width')
    def _change_pen(self):
        for curve in self._curves:
            curve.setPen(pg.mkPen(self.color, width=self.width))

    @on_trait_change('visible')
    def _change_visibility(self):
        self.toggle_visible(status=self.visible)

    def toggle_visible(self, status=None):
        if not self._curves:
            return
        if status is None:
            status = not self._curves[0].isVisible()
        for curve in self._curves:
            curve.setVisible(status)
        self.trait_setq(visible=status)
        return status


class MeanTrace(NavigatorBuilder):
    """This does not really compute, but just grabs the existing mean curve"""

    def set_curve(self, curve):
        self._curves = (curve,)

    def compute(self):
        return self._curves


class StdevTrace(NavigatorBuilder):

    def compute(self):
        if len(self._curves):
            return self._curves
        stdev_fn = partial(np.std, axis=0)
        mean_fn = partial(np.mean, axis=0)
        mu = h5stat(self._array, mean_fn, rowmask=self._rowmask)
        stdev = h5stat(self._array, stdev_fn, rowmask=self._rowmask)
        # now make pg.PlotCurveItems in the navigator plot and set visibility to true
        t = np.arange(len(stdev)) * self._x_scale
        curve1 = pg.PlotCurveItem(x=t, y=(mu - stdev) * self._y_scale)
        curve2 = pg.PlotCurveItem(x=t, y=(mu + stdev) * self._y_scale)
        self._curves = (curve1, curve2)
        self._change_pen()
        return self._curves


class PercentileTrace(NavigatorBuilder):
    percentiles = Str('10, 90', info_text='Percentiles (comma-sep)')

    def compute(self):
        if len(self._curves):
            return self._curves
        pcts = map(float, self.percentiles.split(','))
        curves = []
        for p in pcts:
            def p_fun(x): return np.percentile(x, p, axis=0)
            v = h5stat(self._array, p_fun, rowmask=self._rowmask)
            t = np.arange(len(v)) * self._x_scale
            c = pg.PlotCurveItem(x=t, y=v * self._y_scale)
            curves.append(c)
        self._curves = tuple(curves)
        self._change_pen()
        return self._curves


class NavigatorHandler(Handler):

    def object_name_changed(self, info):
        name = info.object.name.lower()
        if name == 'stdev':
            info.object.nav_menu = StdevTrace(*info.object._nav_args)
        elif name == 'percentiles':
            info.object.nav_menu = PercentileTrace(*info.object._nav_args)


navigator_types = ('Stdev', 'Percentiles')


class NavigatorTrace(HasTraits):
    name = Str
    nav_menu = Instance(NavigatorBuilder)

    def __init__(self, array, x_scale, y_scale, rowmask, **traits):
        self._nav_args = (array, x_scale, y_scale, rowmask)
        super(NavigatorTrace, self).__init__(**traits)

    view = View(
        HGroup(
            UItem('name', style='readonly'),
            Group(UItem('nav_menu', style='custom'), label='Param menu')
        ),
        handler=NavigatorHandler,
    )


trace_tabs = ListEditor(
    use_notebook=True,
    dock_style='tab',
    page_name='.name',
    selected='selected'
)


class NavigatorManager(VisModule):
    name = 'Navigators'
    navigators = List(NavigatorTrace)
    selected = Instance(NavigatorTrace)
    nav_type = Enum(navigator_types[0], navigator_types)
    add_navigator = Button('Add navigator')
    draw_navigator = Button('Draw navigator')

    def __init__(self, **traits):
        super(NavigatorManager, self).__init__(**traits)
        self._data_array = self.parent._qtwindow.array.file_array
        data_channels = self.parent._qtwindow.load_channels
        self.row_mask = np.zeros(self._data_array.shape[0], '?')
        self.row_mask[data_channels] = True
        self.add_navigator_by_type('mean')
        mean_nav = self.navigators[0]
        mean_nav.nav_menu = MeanTrace(*mean_nav._nav_args)
        mean_nav.nav_menu.set_curve(self.parent._qtwindow.nav_trace)
        mean_nav.nav_menu.visible = True
        self.selected = mean_nav

    def add_navigator_by_type(self, nav_type):
        nav = NavigatorTrace(
            self._data_array,
            self.parent.x_scale,
            self.parent._qtwindow.y_scale,
            self.row_mask,
            name=nav_type
        )
        self.navigators.append(nav)
        # This ensures something is selected when there's only one tab
        if len(self.navigators) == 1:
            self.selected = nav

    def _add_navigator_fired(self):
        self.add_navigator_by_type(self.nav_type)

    def _draw_navigator_fired(self):
        nav_builder = self.selected.nav_menu
        curves = nav_builder.compute()
        for c in curves:
            if c not in self.parent._qtwindow.p2.items:
                self.parent._qtwindow.p2.addItem(c)
        nav_builder.visible = True

    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    UItem('nav_type'),
                    UItem('add_navigator'),
                    UItem('draw_navigator', enabled_when='len(selected.nav_menu._curves) == 0')
                ),
                Group(
                    UItem('navigators', style='custom', editor=trace_tabs),
                    show_border=True
                ),
                springy=True
            ),
            title=self.name
        )
        return v
