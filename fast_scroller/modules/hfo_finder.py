from collections import OrderedDict
import numpy as np
import pandas as pd
from traits.api import Int, Float, Range, Button, Bool, Str, File, HasTraits, on_trait_change
from traitsui.api import View, Group, HGroup, VGroup, UItem, Item, Label, RangeEditor

from .base import VisModule
from ..h5scroller import PlotCurveCollection

__all__ = ['HFOLocator']


class HFOCollection(PlotCurveCollection):
    events_table: OrderedDict

    @classmethod
    def cloned_from_collection(cls, curve_collection: PlotCurveCollection, events_table: OrderedDict):
        hfos = cls(curve_collection.hdf_data, curve_collection.plot_channels,
                   curve_collection.dx, curve_collection.dy, curve_collection._y_offset)
        hfos.events_table = events_table
        hfos.setPen(width=4, color='r')
        hfos.setShadowPen(color='r', width=4)
        connection_id = curve_collection.plot_changed.connect(hfos._yoke_offset)
        hfos._yoke = (curve_collection.plot_changed, connection_id)
        return hfos

    def _yoke_offset(self, curves):
        if curves._y_offset != self._y_offset:
            self.y_offset = curves._y_offset

    def __del__(self):
        signal, connection = self._yoke
        signal.disconnect(connection)

    def updatePlotData(self, data_ready=False):
        r = self._view_range()
        if not r:
            print('no view range')
            return
        start, stop = r
        offsets = self.y_offset.squeeze()
        x_data = list()
        y_data = list()
        connections = list()
        for time in self.events_table:
            if start < time < stop:
                e_start = time
                for channel, e_stop in self.events_table[time]:
                    data_channel = self.plot_channels[channel]
                    data = self.hdf_data[data_channel, e_start:e_stop] * self.dy + offsets[channel]
                    y_data.append(data)
                    x_data.append(np.arange(e_start, e_stop) * self.dx)
                    c = np.ones(e_stop - e_start, dtype='>i4')
                    c[-1] = 0
                    connections.append(c)
        print(x_data)
        print(y_data)
        x = np.concatenate(x_data)
        y = np.concatenate(y_data)
        connect = np.concatenate(connections)
        # print('x, y, c =', x.shape, y.shape, connect.shape)
        self.setData(x=x, y=y, connect=connect)


class HFOLocator(VisModule):
    name = Str('HFO Locator')
    file = File(exists=True)
    next = Button('Next event')
    prev = Button('Previous event')
    show_hfos = Bool(True)
    has_events = Bool(False)
    event_idx = Int(-1)
    last_idx = Int(0)
    event_table = OrderedDict()
    # window time in ms
    before_time = Range(low=-1000, high=0, value=-100)
    after_time = Range(low=0, high=1000, value=100)

    @on_trait_change('file')
    def _setup_events(self):
        df = pd.read_csv(self.file, header=None, names=['channel', 'start', 'end'])
        starts = df.start.unique()
        # build a table of all (channel, end-point) pairs for each event start time
        self.event_table = OrderedDict()
        for start in starts:
            df_s = df[df.start == start]
            self.event_table[int(start)] = list(zip(df_s.channel.values, df_s.end.values))
        self.event_idx = -1
        self.last_idx = len(self.event_table)
        self.has_events = True
        self._toggle_curves()

    @on_trait_change('event_idx')
    def move_to_event(self, event=None):
        if event is not None:
            self.trait_setq(event_idx = event)
        # ignore event if event_idx is indeterminate
        if self.event_idx < 0:
            return
        event_time = list(self.event_table.keys())[self.event_idx]
        # get channel, endpoint pairs
        event_data = self.event_table[event_time]
        # covert to actual time
        event_time = event_time * self.parent.x_scale
        view_range = [event_time + self.before_time / 1e3, event_time + self.after_time / 1e3]
        self.parent._qtwindow.update_region(view_range)
        xrange = self.parent._qtwindow.p2.viewRange()[0]
        dx = xrange[1] - xrange[0]
        self.parent._qtwindow.p2.setXRange(event_time - dx / 2, event_time + dx / 2)

    @on_trait_change('show_hfos')
    def _toggle_curves(self):
        if not self.has_events:
            return
        if self.show_hfos:
            self.hfo_curves = HFOCollection.cloned_from_collection(self.curve_collection, self.event_table)
            self.parent._qtwindow.p1.addItem(self.hfo_curves)
        else:
            if not hasattr(self, 'hfo_curves'):
                return
            self.parent._qtwindow.p1.removeItem(self.hfo_curves)
            del self.hfo_curves

    def _change_event(self, sign):
        if 0 <= self.event_idx + sign < self.last_idx:
            self.event_idx += sign

    def _next_fired(self):
        self._change_event(1)

    def _prev_fired(self):
        self._change_event(-1)

    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(Label('Times before/after event (ms)'),
                       UItem('before_time'),
                       UItem('after_time')),
                VGroup(Label('Event file'),
                       UItem('file'),
                       Item('has_events', label='Events ready', style='readonly'),
                       HGroup(Label('Highlight HFOs'), UItem('show_hfos'))),
                VGroup(UItem('next', enabled_when='event_idx < last_idx - 1'),
                       UItem('prev', enabled_when='event_idx > 0'))
            ),
        )
        return v
