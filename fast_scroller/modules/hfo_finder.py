from collections import OrderedDict
import numpy as np
import pandas as pd
from traits.api import Int, Float, Range, Button, Bool, Str, File, HasTraits, on_trait_change
from traitsui.api import View, Group, HGroup, VGroup, UItem, Item, Label, RangeEditor

from .base import VisModule

__all__ = ['HFOLocator']


class HFOLocator(VisModule):
    name = Str('HFO Locator')
    file = File(exists=True)
    next = Button('Next event')
    prev = Button('Previous event')
    clear = Bool('Clear marks')
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
        xrange = self.parent._qtwindow.p2.getXRange()
        dx = xrange[1] - xrange[0]
        self.parent._qtwindow.p2.setXRange([event_time - dx / 2, event_time + dx / 2])

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
                       Item('has_events', label='Events ready', style='readonly')),
                VGroup(UItem('next', enabled_when='event_idx < last_idx - 1'),
                       UItem('prev', enabled_when='event_idx > 0'))
            ),
        )
        return v
