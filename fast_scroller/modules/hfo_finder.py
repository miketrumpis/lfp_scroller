from collections import OrderedDict
import numpy as np
import pandas as pd
from traits.api import Int, Array, Range, Button, Bool, Str, File, on_trait_change
from traitsui.api import View, HGroup, VGroup, UItem, Label, TabularEditor
from traitsui.tabular_adapter import TabularAdapter

from .base import VisModule
from ..h5scroller import PlotCurveCollection, FollowerCollection

__all__ = ['HFOLocator']


class HFOCollection(FollowerCollection):

    def __init__(self, curve_collection: PlotCurveCollection, hfo_lookup: OrderedDict):
        super().__init__(curve_collection)
        self.hfo_lookup = hfo_lookup
        self.setPen(width=4, color='r')
        self.setShadowPen(color='r', width=4)

    # TODO: this version should be functional, but as a subclass, it does break the x-/y-visible contract

    def updatePlotData(self, data_ready=False):
        if not self.can_update:
            return
        r = self._view_range()
        if not r:
            print('no view range')
            return
        start, stop = r
        offsets = self.y_offset.squeeze()
        x_data = list()
        y_data = list()
        visible_size = self.y_visible.shape[1]
        connections = list()
        dx = self.x_visible[0, 1] - self.x_visible[0, 0]
        for time in self.hfo_lookup:
            # convert time in milliseconds to sample index
            t_idx = int(time / 1000 / dx)
            if start < t_idx < stop:
                e_start = t_idx - start
                for channel, e_stop in self.hfo_lookup[time]:
                    e_stop = int(e_stop / 1000 / dx)
                    e_stop = min(visible_size, e_stop - start)
                    plot_channel = self.plot_channels.index(channel)
                    print('slicing ch {} from {}-{} on visible slab'.format(plot_channel, e_start, e_stop))
                    data = self.y_visible[plot_channel, e_start:e_stop] + offsets[plot_channel]
                    y_data.append(data)
                    x_data.append(np.arange(e_start + start, e_stop + start) * self.dx)
                    c = np.ones(e_stop - e_start, dtype='>i4')
                    c[-1] = 0
                    connections.append(c)
        x = np.concatenate(x_data)
        y = np.concatenate(y_data)
        connect = np.concatenate(connections)
        self.setData(x=x, y=y, connect=connect)


class ArrayAdapter(TabularAdapter):

    columns = [('start (ms)', 0), ('end (ms)', 1), ('data channel', 2)]
    # Font fails with wx in OSX; see traitsui issue #13:
    # font        = Font('Courier 10')
    alignment = 'left'
    format = '%d'

    # Could use this if times are floating point
    # def get_format(self, object, trait, row, column):
    #     column_name = self.column_map[column]
    #     if column_name == 'channel':
    #         return '%d'
    #     else:
    #         return '%0.2f'


class HFOLocator(VisModule):
    name = Str('HFO Locator')
    file = File(exists=True)
    next = Button('Next event')
    prev = Button('Previous event')
    show_hfos = Bool(True)
    has_events = Bool(False)
    # Does the events table have 1-based indexing?
    matlab_indexing = Bool(True)
    # Are the channel numbers referring to recording or array channels?
    array_channels = Bool(False)
    event_idx = Int(-1)
    last_idx = Int(0)
    # event_table = Instance(pd.DataFrame)
    event_table = Array
    # tab_select = Any
    hfo_lookup = OrderedDict()
    # window time in ms
    before_time = Range(low=-1000.0, high=0.0, value=-100.0)
    after_time = Range(low=0.0, high=1000.0, value=100.0)

    @on_trait_change('file')
    def _setup_events(self):
        df = pd.read_csv(self.file, header=None, names=['channel', 'start', 'end'])
        df = df.reindex(['start', 'end', 'channel'], axis=1)
        df.channel = df.channel - int(self.matlab_indexing)
        # keep start / end points as times rounded to nearest millisecond
        df.start = np.round((df.start - int(self.matlab_indexing)) * 1000 * self.parent.x_scale).astype('i')
        df.end = np.round((df.end - int(self.matlab_indexing)) * 1000 * self.parent.x_scale).astype('i')
        # If the data channels are in array order, then try to infer what the channels
        # were in recording order. This should be possible by sorting the plot channels
        # and then choosing those channels by their order in the range of array channels.
        if self.array_channels:
            array_rec_order = np.array(sorted(self.curve_collection.plot_channels))
            recording_channels = array_rec_order[df.channel.values]
            df.channel[:] = recording_channels
        df = df.sort_values(by='start', inplace=False).reset_index(drop=True)
        # build a table of all (channel, end-point) pairs for each event start time
        self.hfo_lookup = OrderedDict()

        for start in df.start:
            if int(start) in self.hfo_lookup:
                continue
            df_s = df[df.start == start]
            self.hfo_lookup[int(start)] = list(zip(df_s.channel.values, df_s.end.values))
        self.event_idx = -1
        self.last_idx = len(self.hfo_lookup)
        # convert to regular ndarray because DataFrameEditor is broken :(
        self.event_table = np.c_[df.start, df.end, df.channel]
        self.has_events = True
        self._toggle_curves()

    @on_trait_change('event_idx')
    def move_to_event(self, event=None):
        if event is not None:
            self.trait_setq(event_idx = event)
        # ignore event if event_idx is indeterminate
        if self.event_idx < 0:
            return
        event_time = self.event_table[self.event_idx, 0]
        # covert to actual time
        event_time = float(event_time) / 1000
        print('Event', self.event_idx, end=': ')
        print('Centering on event time', event_time, 'seconds, -/+', self.before_time, self.after_time, 'ms')
        view_range = [event_time + self.before_time / 1e3, event_time + self.after_time / 1e3]
        self.parent._qtwindow.update_region(view_range)
        xrange = self.parent._qtwindow.p2.viewRange()[0]
        dx = xrange[1] - xrange[0]
        self.parent._qtwindow.p2.setXRange(event_time - dx / 2, event_time + dx / 2, padding=0)
        print('Range set: {:.2f}-{:.2f} (dx={:.2f})'.format(event_time - dx / 2, event_time + dx / 2, dx), end=', ')
        xrange = self.parent._qtwindow.p2.viewRange()[0]
        print('Range is: {:.2f}-{:.2f} (dx={:.2f})'.format(xrange[0], xrange[1], xrange[1] - xrange[0]))

    @on_trait_change('show_hfos')
    def _toggle_curves(self):
        if not self.has_events:
            return
        if self.show_hfos:
            self.hfo_curves = HFOCollection(self.curve_collection, self.hfo_lookup)
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
        if self.event_table is None:
            self.event_table = np.array([])
        event_editor = TabularEditor(adapter=ArrayAdapter(),
                                     editable=False,
                                     stretch_last_section=False,
                                     selected_row='event_idx',
                                     scroll_to_row='event_idx')
        v = View(
            HGroup(
                VGroup(Label('Times before/after event (ms)'),
                       UItem('before_time'),
                       UItem('after_time')),
                VGroup(Label('Event file'),
                       UItem('file'),
                       HGroup(UItem('matlab_indexing'), Label('1-based indexing?')),
                       # HGroup(UItem('array_channels'), Label('Array channel numbering?')),
                       HGroup(UItem('show_hfos'), Label('Highlight HFOs'))),
                VGroup(
                       UItem('event_table', editor=event_editor),
                       HGroup(
                           UItem('next', enabled_when='event_idx < last_idx - 1'),
                           UItem('prev', enabled_when='event_idx > 0'))
                       )
            ),
            resizable=True
        )
        return v
