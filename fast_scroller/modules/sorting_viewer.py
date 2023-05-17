import os
import numpy as np
import colorcet
from pickle import load as load_pk
from json import load as load_json
from traits.api import Str, Directory, Button
from traitsui.api import View, HGroup, VGroup, Item, UItem
from pyqtgraph.Qt import QtCore
from ecogdata.parallel.mproc import parallel_context
import pyqtgraph as pg

from .base import VisModule

unit_colors = colorcet.glasbey_bw_minc_20_minl_30
unit_colors = (255 * np.c_[np.array(unit_colors), np.ones(len(unit_colors))]).astype('i')
info = parallel_context.get_logger().info


class SortingViewer(VisModule):
    name = Str('Sorting Rasters')
    sort_output = Directory
    load = Button('Load sorting')
    remove = Button('Remove sorting')

    def _load_fired(self):
        if not os.path.exists(self.sort_output):
            return
        try:
            sorting = np.load(os.path.join(self.sort_output, 'firings.npz'))
            with open(os.path.join(self.sort_output, 'channel_lookup.pk'), 'rb') as fid:
                unit_to_chan = load_pk(fid)
        except FileNotFoundError:
            sorting = np.load(os.path.join(self.sort_output, 'sorter_output', 'firings.npz'))
            with open(os.path.join(self.sort_output, 'sorter_output', 'channel_lookup.pk'), 'rb') as fid:
                unit_to_chan = load_pk(fid)

        # TODO: NOTE: I got stuck here because the only unit-to-channel lookup is saved by channel id,
        #  e.g. channels 14, 16, 18 for the Spikeinterface dataset, whereas these are only known as channels 3, 4,
        #  5 in the file reader and the curve collection

        # This is EXTREMELY DIRTY but can back-track the sorting channels from the original channels using the
        # locations in the JSON file
        with open(os.path.join(self.sort_output, 'spikeinterface_recording.json'), 'r') as fid:
            recording_json = load_json(fid)

        # This is highly dependent on the sorter recording having this heirarchy: CAR{SLICE{RecordingExtractor}}
        level = recording_json
        while 'ChannelSlice' not in level['class']:
            if 'recording' in level['kwargs']:
                level = level['kwargs']['recording']
            else:
                level = None
                break
        if level is None:
            raise RuntimeError('This sorting did not use a channel slice??')

        # active_channel_locs = recording_json['properties']['location']
        active_channel_locs = level['properties']['location']
        active_channel_ids = level['kwargs']['renamed_channel_ids']
        # original_locs = recording_json['kwargs']['recording']['kwargs']['parent_recording']['properties']['location']
        original_locs = level['kwargs']['parent_recording']['properties']['location']
        chan_to_chan = dict()
        # The curve manager knows what channels are ACTUALLY displayed -- only map unit channels on the scroller screen
        curves = self.parent.curve_manager.source_curve
        for loc, chan in zip(active_channel_locs, active_channel_ids):
            data_order_chan = original_locs.index(loc)
            if data_order_chan in curves.plot_channels:
                chan_to_chan[chan] = data_order_chan

        # a table mapping unit IDs to channel index (but only channels on the screen)
        self.unit_to_chan = dict()
        for u, c in unit_to_chan.items():
            if u in sorting['unit_ids'] and c in chan_to_chan:
                self.unit_to_chan[u] = chan_to_chan[c]
        units_on_screen = self.unit_to_chan.keys()
        units = sorting['spike_labels_seg0']
        u_mask = np.array([u in units_on_screen for u in units])
        self.units = units[u_mask]
        self.spikes = sorting['spike_indexes_seg0'][u_mask]

        # Need to add ScatterPlotItems to the plot
        p1 = self.parent._qtwindow.p1
        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        p1.addItem(self.scatter)
        p1.sigRangeChanged.connect(self.draw_spikes_in_p1_range)
        p1.sigRangeChanged.emit(None, p1.viewRange())

    def draw_spikes_in_p1_range(self, window, vrange):
        # plot = self.parent._qtwindow.p1
        curves = self.parent.curve_manager.source_curve
        i1, i2 = curves.x_to_sample(np.array([vrange[0][0], vrange[0][1]]))
        events = (self.spikes >= i1) & (self.spikes <= i2)
        info('Found {} events from {} to {}'.format(len(events), i1, i2))

        channel_offsets = dict([(chan, curves.channel_offset(chan)) for chan in curves.plot_channels])
        # range_times = self.spikes[events] * self.parent.x_scale
        range_times = curves.sample_to_x(self.spikes[events])
        range_units = self.units[events]
        range_offsets = [channel_offsets[self.unit_to_chan[u]] for u in range_units]
        # range_offsets = np.array(range_offsets) - 0.5 * self.parent.y_spacing / 1e6
        range_colors = [pg.mkBrush(unit_colors[u % len(unit_colors)]) for u in range_units]
        # spots = [dict(pos=(t, v), data=1,
        self.scatter.setData(pos=np.c_[range_times, range_offsets], brush=range_colors)

    def _remove_fired(self):
        self.scatter.setPointsVisible(False)
        p1 = self.parent._qtwindow.p1
        try:
            p1.sigRangeChanged.disconnect(self.draw_spikes_in_p1_range)
        except RuntimeError:
            pass
        vb = p1.getViewBox()
        vb.sigStateChanged.emit(vb)

    def default_traits_view(self):
        v = View(
            HGroup(
                Item('sort_output'),
                VGroup(
                    UItem('load'),
                    UItem('remove')
                )
            )
        )
        return v
