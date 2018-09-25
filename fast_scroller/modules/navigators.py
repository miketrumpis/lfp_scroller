from functools import partial
import numpy as np
import pyqtgraph as pg

from traits.api import Button, Str, on_trait_change, Bool
from traitsui.api import View, HGroup, Group, UItem, Item

from .base import VisModule
from ..h5data import h5stat


__all__ = ['NavigatorManager']


class NavigatorManager(VisModule):
    name = 'Navigators'

    percentiles = Str
    draw_percentiles = Button('Draw')
    percentile_visible = Bool(False)
    _percentile_computed = Bool(False)

    draw_stdev = Button('Draw')
    stdev_visible = Bool(False)
    _stdev_computed = Bool(False)


    def __init__(self, **traits):
        super(NavigatorManager, self).__init__(**traits)
        self._data_array = self.parent._qtwindow.array.file_array
        data_channels = self.parent._qtwindow.load_channels
        self.row_mask = np.zeros(self._data_array.shape[0], '?')
        self.row_mask[data_channels] = True
        print self.row_mask.shape, self.row_mask.dtype
        self._stored_percentiles = dict()
        self._stdev_curves = None


    def set_curve_visible(curve, status=None):
        if status is None:
            status = not curve.isVisible()
        curve.setVisible(status)
        return status


    def _draw_stdev_fired(self):
        # Hitting draw either creates curves or sets them visible
        if self._stdev_curves is not None:
            self.set_curve_visible(self._stdev_curves[0], True)
            self.set_curve_visible(self._stdev_curves[1], True)
            self.trait_setq(stdev_visible=True)
            return
        stdev_fn = partial(np.std, axis=0)
        mean_fn = partial(np.mean, axis=0)
        scale = self.parent._qtwindow.y_scale
        mu = h5stat(self._data_array, mean_fn, rowmask=self.row_mask)
        stdev = h5stat(self._data_array, stdev_fn, rowmask=self.row_mask)
        # now make pg.PlotCurveItems in the navigator plot and set visibility to true
        t = np.arange(len(stdev)) * self.parent.x_scale
        curve1 = pg.PlotCurveItem(x=t, y=(mu - stdev) * scale, pen='r')
        curve2 = pg.PlotCurveItem(x=t, y=(mu + stdev) * scale, pen='r')
        self._stdev_curves = (curve1, curve2)
        self.parent._qtwindow.p2.addItem(curve1)
        self.parent._qtwindow.p2.addItem(curve2)
        self.trait_setq(stdev_visible=True)
        self._stdev_computed = True


    @on_trait_change('stdev_visible')
    def _toggle_stdev(self):
        self.set_curve_visible(self._stdev_curves[0], self.stdev_visible)
        self.set_curve_visible(self._stdev_curves[1], self.stdev_visible)


    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    UItem('draw_stdev'),
                    Item('stdev_visible', label='Visible', enabled_when='_stdev_computed'),
                    label='Stdev margins'
                ),
                Group(
                    UItem('draw_percentiles'),
                    Item('percentile_visible', label='Visible', enabled_when='_percentile_computed'),
                    Item('percentiles', label='Percentiles'),
                    label='Percentile margins'
                ),
                label='Navigation traces'
            ),
        )
        return v



