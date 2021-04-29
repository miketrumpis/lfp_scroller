from collections import OrderedDict
import numpy as np
import pandas as pd
from scaleogram import CWT, cws
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, FuncFormatter
from scipy.signal import detrend
from traits.api import Int, Float, Array, Range, Button, Bool, Str, File, observe, Instance, Tuple
from traitsui.api import View, HGroup, VGroup, UItem, Label, TabularEditor
from traitsui.tabular_adapter import TabularAdapter

from ecogdata.filt.time import slepian_projection
from ecogdata.devices.units import nice_unit_text
from ecoglib.vis import plotters

from .base import PlotsInterval, GridSpecStack
from ..curve_collections import PlotCurveCollection, SelectedFollowerCollection

__all__ = ['HFOLocator']


@FuncFormatter
def _sec_to_msec(sec, pos):
    return '{:d}'.format(int(sec * 1000))


def hfo_plot(lfp: np.ndarray, full_window: tuple, hfo_window: tuple, gridspec: GridSpec,
             hfo_band: tuple=(80, 500),
             normalize: bool=True):
    # times are given and displayed in milliseconds
    start, end = [t / 1000.0 for t in full_window]
    hfo_start, hfo_end = [t / 1000.0 for t in hfo_window]
    win_size = len(lfp)
    dt = (end - start) / win_size
    lfp_flat = detrend(lfp, type='linear')
    hfo_time = np.arange(win_size) * dt + start
    hfo_bw = (hfo_band[1] - hfo_band[0]) / 2
    hfo_center = (hfo_band[0] + hfo_band[1]) / 2
    hfo_filtered = slepian_projection(lfp_flat, hfo_bw, 1 / dt, w0=hfo_center, min_conc=0.96)
    uv_text = nice_unit_text('uv')

    # The wavelet name "cmorB-C" encodes:
    # B for bandwidth (Gaussian width): larger BW --> tighter time concentration
    # C for center freq (cycle rate): higher frequency --> greater number of cycles within Gaussian window
    cwt = CWT(hfo_time,
              signal=lfp_flat,
              scales=np.arange(1, win_size // 2),
              wavelet='cmor1-1'
              # wavelet='cmor1-1.5'
              )
    if normalize:
        n_window = int((hfo_window[0] - start) / dt)
        cwt_mu = np.abs(cwt.coefs[..., :n_window]).mean(axis=1, keepdims=True)
        cwt.coefs /= cwt_mu

    f = gridspec.figure
    ax_full = f.add_subplot(gridspec[0])
    ax_full.plot(hfo_time, lfp)
    ax_full.set_ylabel(u'LFP ' + uv_text)
    ax_hfo = f.add_subplot(gridspec[1], sharex=ax_full)
    ax_hfo.plot(hfo_time, hfo_filtered)
    ax_hfo.set_ylabel(u'HFO ' + uv_text)
    ax_sg = f.add_subplot(gridspec[2], sharex=ax_full)
    cws(cwt,
        # coi=False,
        coikw=dict(alpha=1.0, facecolor='gray'),
        # cbar='horizontal',
        cbar=None,
        cmap='viridis',
        cscale=('linear' if normalize else 'log'),
        yscale='log',
        yaxis='frequency',
        ax=ax_sg)
    ax_sg.set_title('')
    clim = np.percentile(np.abs(cwt.coefs), [1, 99])
    mesh = ax_sg.collections[0]
    mesh.set_clim(*clim)
    if max(ax_sg.get_ylim()) > 1e3:
        ax_sg.set_ylim(top=1e3)
    cb_ax = f.add_subplot(gridspec[3])
    cb = f.colorbar(mesh, cax=cb_ax, orientation='horizontal')
    # cb_ax = [ax for ax in f.axes if 'colorbar' in ax.get_label()]
    t = 'Wavelet magnitude'
    if normalize:
        t = t + ' over baseline'
    cb.set_label(t)
    for ax in (ax_full, ax_hfo):
        plotters.plt.setp(ax.get_xticklabels(), visible=False)
        # ax.xaxis.set_major_formatter(NullFormatter())
        ax.axvspan(hfo_start, hfo_end, facecolor='g', alpha=0.15, zorder=-10)
    ax_sg.set_xlabel('Time (ms)')
    gridspec.tight_layout(f, h_pad=0.5)
    ax_sg.xaxis.set_major_formatter(_sec_to_msec)
    # xticks = ax_sg.get_xticks()
    # print('changing xticks from', xticks)
    # ax_sg.set_xticklabels(['{:d}'.format(int(t * 1000)) for t in xticks])
    return f


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
    alignment = 'left'
    format = '%d'

    # Could use this if times are floating point
    # def get_format(self, object, trait, row, column):
    #     column_name = self.column_map[column]
    #     if column_name == 'channel':
    #         return '%d'
    #     else:
    #         return '%0.2f'


class HFOLocator(PlotsInterval):
    name = Str('HFO Locator')
    # Only plots new figures in gridspec format
    new_figure = Bool(True)
    _figstack = Instance(GridSpecStack, args=())
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
    hfo_band = Tuple(80.0, 500.0)  # Need to specify "N.0" for floats (otherwise will type as int)
    normalize_scalogram = Bool(True)
    plot_hfo = Button('plot HFO')

    @observe('file')
    def _read_table(self, event):
        df = pd.read_csv(self.file, header=None, names=['channel', 'start', 'end'])
        df = df.reindex(['start', 'end', 'channel'], axis=1)
        df.channel = df.channel - int(self.matlab_indexing)
        # keep start / end points as times rounded to nearest millisecond
        offset = self.time_offset
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
        self._toggle_curves(None)

    @observe('event_idx, before_time, after_time')
    def move_to_event(self, event, hfo_idx=None):
        if hfo_idx is not None:
            self.trait_setq(event_idx=hfo_idx)
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

    @observe('show_hfos')
    def _toggle_curves(self, event):
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

    def _plot_hfo_fired(self):
        self.plot_current_hfo()

    def plot_current_hfo(self):
        if not len(self.event_table) or self.event_idx < 0:
            return
        hfo_start, hfo_end, channel = self.event_table[self.event_idx]
        start = hfo_start + self.before_time
        end = hfo_start + self.after_time
        source = self.curve_manager.source_curve
        page_start = int(self.page_limits[0] / source.dx)
        start_idx = int(start / 1000 / source.dx) - page_start
        end_idx = min(source.y_visible.shape[1], int(end / 1000 / source.dx) - page_start)
        source_channel = source.plot_channels.index(channel)
        hfo_segment = source.y_visible[source_channel, start_idx:end_idx] * 1e6
        f, gs = self._get_fig(nrows=4, ncols=1, height_ratios=[1, 1, 3, 0.2], figsize=(4, 6))
        print('Channel', source_channel, 'Full window:', start, end, 'HFO window:', hfo_start, hfo_end)
        f = hfo_plot(hfo_segment, (start, end), (hfo_start, hfo_end), gs,
                     hfo_band=self.hfo_band,
                     normalize=self.normalize_scalogram)
        try:
            f.canvas.draw()
        except:
            pass


    def _time_offset_changed(self):
        if len(self.event_table):
            if hasattr(self, 'hfo_curves'):
                self.curve_manager.remove_curves('HFOs')
                del self.hfo_curves
            if self.file:
                self._read_table(None)

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
                VGroup(Label('Event file'),
                       UItem('file'),
                       HGroup(
                           VGroup(HGroup(UItem('matlab_indexing'), Label('1-based indexing?')),
                                  HGroup(UItem('time_offset'), Label('Offset start time (ms)'))),
                           VGroup(HGroup(UItem('show_hfos'), Label('Highlight HFOs')))
                               # HGroup(UItem('array_channels'), Label('Array channel numbering?')),
                       )),
                VGroup(Label('Times before/after event (ms)'),
                       UItem('before_time'),
                       UItem('after_time'),
                       HGroup(Label('HFO band'), UItem('hfo_band', style='simple')),
                       HGroup(Label('Normalize Scgram'), UItem('normalize_scalogram')),
                       UItem('plot_hfo')),
                VGroup(
                    UItem('event_table', editor=event_editor),
                    HGroup(UItem('next', enabled_when='event_idx < last_idx - 1'),
                           UItem('prev', enabled_when='event_idx > 0'))
                )
            ),
            resizable=True
        )
        return v
