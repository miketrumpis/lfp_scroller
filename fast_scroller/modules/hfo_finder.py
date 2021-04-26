from collections import OrderedDict
import numpy as np
import pandas as pd
from traits.api import Int, Float, Array, Range, Button, Bool, Str, File, on_trait_change
from traitsui.api import View, HGroup, VGroup, UItem, Label, TabularEditor
from traitsui.tabular_adapter import TabularAdapter

from .base import VisModule
from ..curve_collections import PlotCurveCollection, SelectedFollowerCollection

__all__ = ['HFOLocator']


class HFOCollection(SelectedFollowerCollection):

    def __init__(self, curve_collection: PlotCurveCollection, hfo_lookup: OrderedDict):
        super().__init__(curve_collection)
        self.hfo_lookup = hfo_lookup
        self.setPen(width=4, color='r')
        self.setShadowPen(color='r', width=4)
        self._x_padded = None
        self._y_padded = None
        self._compressed_hfos = None

    @property
    def x_visible(self):
        return self._x_padded

    @x_visible.setter
    def x_visible(self, new):
        return

    @property
    def y_visible(self):
        return self._y_padded

    @y_visible.setter
    def y_visible(self, new):
        return

    def _set_hfo_data(self):
        r = self._view_range()
        if not r:
            return
        start, stop = r
        full_page_size = stop - start
        active_channels = np.zeros(len(self._active_channels), '?')
        y_padded = np.empty(self._source.y_visible.shape)
        y_padded.fill(np.nan)
        x_padded = np.empty(y_padded.shape)
        x_padded.fill(np.nan)
        # offset = self._y_offset
        x_data = list()
        y_data = list()
        # save the number of y steps (apply actual units offset based on state)
        y_step = list()
        connections = list()
        for time in self.hfo_lookup:
            # convert time in milliseconds to sample index
            t_idx = int(time / 1000 / self.dx)
            if start < t_idx < stop:
                e_start = t_idx - start
                for channel, e_stop in self.hfo_lookup[time]:
                    e_stop = int(e_stop / 1000 / self.dx)
                    e_stop = min(full_page_size, e_stop - start)
                    plot_channel = self._source.plot_channels.index(channel)
                    active_channels[plot_channel] = True
                    print('slicing ch {} from {}-{} on visible slab'.format(plot_channel, e_start, e_stop))
                    # Compact sets for the plot data
                    data = self._source.y_visible[plot_channel, e_start:e_stop]
                    y_data.append(data)
                    y_step.append(plot_channel)
                    x_data.append(np.arange(e_start + start, e_stop + start) * self.dx)
                    c = np.ones(e_stop - e_start, dtype='>i4')
                    c[-1] = 0
                    connections.append(c)
                    # Uncompressed (nan-filled) sets for the x-/y-visible arrays
                    y_padded[plot_channel, e_start:e_stop] = self._source.y_visible[plot_channel, e_start:e_stop]
                    x_padded[plot_channel, e_start:e_stop] = self._source.x_visible[plot_channel, e_start:e_stop]
        self._x_padded = x_padded[active_channels]
        self._y_padded = y_padded[active_channels]
        self._compressed_hfos = (np.concatenate(x_data), np.concatenate(connections), y_data, y_step)
        print(len(self._compressed_hfos))
        self._active_channels = active_channels

    def _source_triggered_update(self, source):
        self._set_hfo_data()
        super()._source_triggered_update(source)

    def updatePlotData(self, data_ready=False):
        if not self.can_update:
            return
        r = self._view_range()
        if not r:
            print('no view range')
            return
        x, connect = self._compressed_hfos[:2]
        y_data, y_step = self._compressed_hfos[2:]
        if not len(x):
            return
        # the y-data needs to be assembled depending on the present y offset
        y = np.concatenate([y + n * self._y_offset for y, n in zip(y_data, y_step)])
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
    # Does the events table have 1-based indexing?
    matlab_indexing = Bool(True)
    # Are the channel numbers referring to recording or array channels?
    array_channels = Bool(False)
    time_offset = Float(0)
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
    def _read_table(self):
        df = pd.read_csv(self.file, header=None, names=['channel', 'start', 'end'])
        df = df.reindex(['start', 'end', 'channel'], axis=1)
        df.channel = df.channel - int(self.matlab_indexing)
        # keep start / end points as times rounded to nearest millisecond
        offset = self.time_offset * 1000
        df.start = np.round(df.start + offset).astype('i')
        df.end = np.round(df.end + offset).astype('i')
        # If the data channels are in array order, then try to infer what the channels
        # were in recording order. This should be possible by sorting the plot channels
        # and then choosing those channels by their order in the range of array channels.
        if self.array_channels:
            array_rec_order = np.array(sorted(self.curve_manager.source_curve.plot_channels))
            recording_channels = array_rec_order[df.channel.values]
            df.channel[:] = recording_channels
        df = df.sort_values(by='start', inplace=False).reset_index(drop=True)
        self.event_table = np.c_[df.start, df.end, df.channel]
        self.create_event_lookup(self.event_table)

    def create_event_lookup(self, table):
        start_times, end_times, channels = table.transpose()
        # build a table of all (channel, end-point) pairs for each event start time
        self.hfo_lookup = OrderedDict()

        for start in start_times:
            if int(start) in self.hfo_lookup:
                continue
            mask = start_times == start
            self.hfo_lookup[int(start)] = list(zip(channels[mask], end_times[mask]))
        self.event_idx = -1
        # self.last_idx = len(self.hfo_lookup)
        self.last_idx = len(start_times)
        # convert to regular ndarray because DataFrameEditor is broken :(
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
        if not len(self.event_table):
            return
        if self.show_hfos:
            self.hfo_curves = HFOCollection(self.curve_manager.source_curve, self.hfo_lookup)
            self.curve_manager.add_new_curves(self.hfo_curves, 'HFOs')
        else:
            if not hasattr(self, 'hfo_curves'):
                return
            self.curve_manager.remove_curves('HFOs')
            del self.hfo_curves

    def _time_offset_changed(self):
        if len(self.event_table):
            if hasattr(self, 'hfo_curves'):
                self.curve_manager.remove_curves('HFOs')
                del self.hfo_curves
            if self.file:
                self._read_table()

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
                       HGroup(
                           VGroup(HGroup(UItem('matlab_indexing'), Label('1-based indexing?')),
                                  HGroup(UItem('time_offset'), Label('Offset start time (s)'))),
                           VGroup(HGroup(UItem('show_hfos'), Label('Highlight HFOs')))
                               # HGroup(UItem('array_channels'), Label('Array channel numbering?')),
                       )),
                VGroup(
                    UItem('event_table', editor=event_editor),
                    HGroup(UItem('next', enabled_when='event_idx < last_idx - 1'),
                           UItem('prev', enabled_when='event_idx > 0'))
                )
            ),
            resizable=True
        )
        return v
