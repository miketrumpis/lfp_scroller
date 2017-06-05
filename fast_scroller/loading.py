#!/usr/bin/env python
"""
Objects for loading/filtering HDF5 data streams and launching the
pyqtgraph / traitsui vis tool.
"""

from __future__ import division
import os
from functools import partial
from tempfile import NamedTemporaryFile
import numpy as np
from scipy.signal import detrend
import h5py

# these imports set up some qt runtime stuff (??)
import PySide
import pyqtgraph as pg

from traits.api import Instance, Button, HasTraits, File, Float, \
     Str, Property, List, Enum, Int, Bool, on_trait_change
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     Label, Handler, EnumEditor, Group, TableEditor, Tabbed, spring, \
     FileEditor, SetEditor
from traitsui.table_column import ObjectColumn

from ecogana.devices.electrode_pinouts import get_electrode_map, electrode_maps
from ecogana.devices.load.open_ephys import hdf5_open_ephys_channels
from ecogana.devices.channel_picker import interactive_mask
from ecoglib.filt.time.design import butter_bp, cheby1_bp, cheby2_bp, notch
from ecoglib.util import ChannelMap, Bunch

from h5scroller import FastScroller
from h5data import bfilter, FilteredReadCache, h5mean, ReadCache, \
     DCOffsetReadCache, CommonReferenceReadCache

from new_scroller import VisWrapper
from .modules import ana_modules, default_modules

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

class FileHandler(Handler):

    fields = List(Str)
    def object_file_changed(self, info):
        if not len(info.object.file):
            self.fields = []
            return
        h5 = h5py.File(info.object.file, 'r')
        self.fields = h5.keys()
        h5.close()

class FileData(HasTraits):
    """Basic model for an HDF5 file holding an array timeseries."""
    file = File(filter=[u'*.h5'])
    data_field = Str
    fs_field = Str
    y_scale = Float(1.0)
    chan_map = Enum(sorted(electrode_maps.keys())[0],
                    sorted(electrode_maps.keys()))
    zero_windows = Bool
    car_windows = Bool

    def _compose_arrays(self, filter, rowmask=(), colmask=()):
        # default is just a wrapped array
        h5 = h5py.File(self.file, 'r')
        array = filter(h5[self.data_field])
        mn_trace = h5mean(array, 0, rowmask=rowmask, colmask=colmask)
        mn_trace *= self.y_scale
        if self.zero_windows:
            array = FilteredReadCache(array, partial(detrend, type='constant'))
        elif self.car_windows:
            array = CommonReferenceReadCache(array)
        else:
            array = ReadCache(array)
        return array, mn_trace
    
    def default_traits_view(self):
        view = View(
            VGroup(
                Item('file', label='Visualize this file'),
                HGroup(
                    Item('data_field', label='Data field',
                         editor=EnumEditor(name='handler.fields')),
                    Item('fs_field', label='Fs field',
                         editor=EnumEditor(name='handler.fields'))
                    ),
                Label('Scales samples to voltage (mV)'),
                UItem('y_scale'),
                Item('chan_map', label='channel map name'),
                HGroup(
                    Item('zero_windows', label='Remove local DC?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    )
                ),
            handler=FileHandler,
            resizable=True,
            title='Launch Visualization'
        )
        return view

class OpenEphysFileData(FileData):
    file = File(filter=[u'*.h5', u'*.continuous'])
    data_field = Property(fget=lambda self: 'chdata')
    fs_field = Property(fget=lambda self: 'Fs')
    y_scale = Float(0.001)

    # convert controls
    can_convert = Property(
        fget=lambda self: os.path.splitext(self.file)[1] == '.continuous'
        )
    temp_conv = Bool(False)
    conv_file = File
    downsamp = Int(1)
    quantized = Bool(False)
    do_convert = Button('Run convert')
    

    @on_trait_change('file')
    def _sync_files(self):
        if self.can_convert:
            rec, _ = os.path.split(self.file)        
            self.conv_file = rec + '.h5'

    def _do_convert_fired(self):
        # check extension
        if os.path.splitext(self.file) == '.h5':
            print 'already an HDF5 file!!'
            return
        # self.file is '/exp/rec/file.continuous'
        rec, _ = os.path.split(self.file)
        exp, rec = os.path.split(rec)
        if self.temp_conv:
            if not self.conv_file:
                ev = Error(
                    error_msg='Temp file not implemented, set a convert file'
                    )
                ev.edit_traits()
                return
            self.temp_conv = False
        if self.temp_conv:
            with NamedTemporaryFile(mode='ab') as h5file:
                h5file.close()
                hdf5_open_ephys_channels(
                    exp, rec, h5file.name,
                    downsamp=self.downsamp, quantized=self.quantized
                    )
            self.file = self.h5file.name
        else:
            if not self.conv_file:
                ev = Error(
                    error_msg='Set a convert file first'
                    )
                ev.edit_traits()
                return
            print 'converting to', self.conv_file
            hdf5_open_ephys_channels(
                exp, rec, self.conv_file,
                downsamp=self.downsamp, quantized=self.quantized
                )
            self.file = self.conv_file
                        
    def default_traits_view(self):
        v = View(
            VGroup(
                Item('file', label='Visualize this file'),
                VGroup(
                    Item('conv_file',
                         editor=FileEditor(dialog_style='save'),
                         label='HDF5 convert file'),
                    HGroup(
                        Item('temp_conv', label='Use temp file'),
                        Item('downsamp', label='Downsample factor'),
                        Item('quantized', label='Quantized?'),
                        label='Conversion options'
                        ),
                    UItem('do_convert'),
                    enabled_when='can_convert'
                    ),
                    
                Label('Scales samples to mV'),
                Label('(1.98e-4 if quantized, else 1e-3)'),
                UItem('y_scale'),
                Item('chan_map', label='channel map name'),
                HGroup(
                    Item('zero_windows', label='Remove local DC?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    )
                ),
            resizable=True,
            title='Launch Visualization'
        )
        return v
        
    
class Mux7FileData(FileData):
    """
    An HDF5 array holding timeseries from the MUX headstages:
    defaults to building an ReadCache with DC offset subraction.
    """

    y_scale = Property(fget=lambda self: 1e3 / self.gain)
    gain = Enum( 12, (3, 10, 12, 20) )
    chan_map = Str('ratv4_mux6')
    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    
    def _compose_arrays(self, filter, rowmask=(), colmask=()):
        if self.zero_windows or self.car_windows:
            return super(Mux7FileData, self)._compose_arrays(filter)
        f = h5py.File(self.file, 'r')
        array = filter(f[self.data_field])
        mn_level = h5mean(array, 1, start=int(3e4))
        mn_trace = h5mean(array, 0, rowmask=rowmask, colmask=colmask)
        mn_trace *= self.y_scale
        array = DCOffsetReadCache(array, mn_level)
        return array, mn_trace

    def default_traits_view(self):
        view = View(
            VGroup(
                Item('file', label='Visualize this file'),
                Label('Amplifier gain (default 12 for Mux v7)'),
                UItem('gain'),
                Item('chan_map', label='channel map name (use default)'),
                HGroup(
                    Item('zero_windows', label='Remove local DC?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    )
                ),
        )
        return view

class FilterMenu(HasTraits):
    """Edits filter arguments."""
    def default_traits_view(self):
        # builds a panel automatically from traits
        t = self.traits()
        items = list()
        for k, v in t.items():
            if k in ('trait_added', 'trait_modified'):
                continue
            items.extend( [Label(v.info_text), UItem(k, style='simple')] )
        return View(*items, resizable=True)
    
class BandpassMenu(FilterMenu):
    lo = Float(-1, info_text='low freq cutoff (-1 for none)')
    hi = Float(-1, info_text='high freq cutoff (-1 for none)')
    ord = Int(5, info_text='filter order')

class ButterMenu(BandpassMenu):
    def make_filter(self, Fs):
        return butter_bp(lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord)

class Cheby1Menu(BandpassMenu):
    ripple = Float(0.1, info_text='passband ripple (dB)')
    def make_filter(self, Fs):
        return cheby1_bp(
            self.ripple, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
            )

class Cheby2Menu(BandpassMenu):
    stop_atten = Float(40, info_text='stopband attenuation (dB)')
    def make_filter(self, Fs):
        return cheby2_bp(
            self.ripple, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
            )
    
class NotchMenu(FilterMenu):
    notch = Float(60, info_text='notch frequency')
    nwid = Float(3.0, info_text='notch width (Hz)')
    nzo = Int(3, info_text='number of zeros at notch')
    def make_filter(self, Fs):
        return notch(self.notch, Fs, self.nwid, nzo=self.nzo)
    
class FilterHandler(Handler):

    def object_filt_type_changed(self, info):
        # check whether there is already a menu of the
        # correct type -- this method appears to be called
        # whenever a new handler is created!
        if info.object.filt_type == 'butterworth':
            if not isinstance(info.object.filt_menu, ButterMenu):
                info.object.filt_menu = ButterMenu()
        elif info.object.filt_type == 'cheby 1':
            if not isinstance(info.object.filt_menu, Cheby1Menu):
                info.object.filt_menu = Cheby1Menu()
        elif info.object.filt_type == 'cheby 2':
            if not isinstance(info.object.filt_menu, Cheby2Menu):
                info.object.filt_menu = Cheby2Menu()
        elif info.object.filt_type == 'notch':
            if not isinstance(info.object.filt_menu, NotchMenu):
                info.object.filt_menu = NotchMenu()


class Filter(HasTraits):
    """Filter model with adaptive menu. Menu can build transfer functions."""
    
    filt_type = Enum( 'butterworth', ['butterworth', 'cheby 1',
                                      'cheby 2', 'notch'] )
    filt_menu = Instance(HasTraits)

    view = View(
        VGroup(
            Item('filt_type', label='Filter type'),
            Group(
                UItem('filt_menu', style='custom'),
                label='Filter menu'
                )
            ),
        handler=FilterHandler
        )
    
filter_table = TableEditor(
    columns = [ ObjectColumn( name='filt_type' ) ],
    deletable=True,
    auto_size=True,
    show_toolbar=True,
    reorderable=True,
    edit_view=None,
    row_factory=Filter
    )

def _pipeline_factory(f_list):
    if not len(f_list):
        return lambda x: x
    def copy_x(x):
        with NamedTemporaryFile(mode='ab', dir='.') as f:
            print f.name
            f.file.close()
            fw = h5py.File(f.name, 'w')
            y = fw.create_dataset('data', shape=x.shape,
                                  dtype=x.dtype, chunks=x.chunks)
        return y
        
    def run_list(x, axis=1):
        y = copy_x(x)
        b, a = f_list[0]
        bfilter(b, a, x, axis=axis, out=y)
        for (b, a) in f_list[1:]:
            bfilter(b, a, y, axis=axis)
        return y

    return run_list
        

class FilterPipeline(HasTraits):
    """List of filters that can be pipelined."""
    filters = List(Filter)

    def make_pipeline(self, Fs):
        filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
        return _pipeline_factory(filters)
    
    view = View(
        Group(
            UItem('filters', editor=filter_table),
            show_border=True
            ),
        title='Filters',
        )

class HeadstageHandler(Handler):

    def object_headstage_changed(self, info):
        if not info.initialized and info.object.file_data:
            return
        hs = info.object.headstage
        if hs.lower() == 'mux3':
            fd = Mux7FileData(gain=10)
        elif hs.lower() in ('mux5', 'mux6'):
            fd = Mux7FileData(gain=20)
        elif hs.lower() == 'mux7':
            fd = Mux7FileData(gain=12)
        elif hs.lower() == 'intan':
            fd = OpenEphysFileData()
        else:
            fd = FileData()
        info.object.file_data = fd

class VisLauncher(HasTraits):
    """
    Builds pyqtgraph/traitsui visualation using a timeseries
    from a FileData model that is filtered from a FilterPipeline.
    """
    
    file_data = Instance(FileData)
    filters = Instance(FilterPipeline)
    module_set = List(default_modules)
    all_modules = List(ana_modules.keys())
    b = Button('Launch Visualization')
    offset = Enum(0.2, [0.1, 0.2, 0.5, 1, 2, 5])
    max_window_width = Float(1200.0)
    headstage = Enum('mux7',
                     ('mux3', 'mux5', 'mux6', 'mux7', 'intan', 'unknown'))
    screen_channels = Bool(False)

    def __init__(self, **traits):
        super(VisLauncher, self).__init__(**traits)
        self.add_trait('filters', FilterPipeline())

    def _get_screen(self, array, channels, chan_map, Fs):
        from ecogana.expconfig import params
        mem_guideline = float(params.memory_limit)
        n_chan = len(array)
        word_size = array.dtype.itemsize
        n_pts = min(100000, mem_guideline / n_chan / word_size)
        data = np.empty( (len(channels), n_pts), dtype=array.dtype )
        for n, c in enumerate(channels):
            data[n] = array[c, :n_pts]

        data_bunch = Bunch(
            data=data, chan_map=chan_map, Fs=Fs, units='au', name=''
            )
        mask = interactive_mask(data_bunch, use_db=False)
        screen_channels = [ channels[i] for i in xrange(len(mask)) if mask[i] ]
        screen_map = chan_map.subset(mask)
        return screen_channels, screen_map
        

    def launch(self):
        if not os.path.exists(self.file_data.file):
            return
        
        chan_map, nc = get_electrode_map(self.file_data.chan_map)
        data_channels = range( len(chan_map) + len(nc) )
        for chan in nc:
            data_channels.remove(chan)

        # permute  channels to stack rows
        chan_idx = zip(*chan_map.to_mat())
        chan_order = chan_map.lookup(*zip( *sorted(chan_idx)[::-1] ))
        data_channels = [data_channels[i] for i in chan_order]
        chan_map = ChannelMap( [chan_map[i] for i in chan_order],
                               chan_map.geometry, col_major=chan_map.col_major )

        h5 = h5py.File(self.file_data.file, 'r')
        x_scale = h5[self.file_data.fs_field].value ** -1.0
        num_vectors = h5[self.file_data.data_field].shape[0]
        h5.close()
        filters = self.filters.make_pipeline(x_scale ** -1.0)
        rm = np.zeros( (num_vectors,), dtype='?' )
        rm[data_channels] = True
        array, nav = self.file_data._compose_arrays(filters, rowmask=rm)
        if self.screen_channels:
            data_channels, chan_map = \
              self._get_screen(array, data_channels, chan_map, x_scale**-1.0)
        modules = [ana_modules[k] for k in self.module_set]
        new_vis = FastScroller(array, self.file_data.y_scale,
                               self.offset, chan_map, nav,
                               x_scale=x_scale,
                               load_channels=data_channels,
                               max_zoom=self.max_window_width)
        v_win = VisWrapper(new_vis, x_scale = x_scale, chan_map=chan_map,
                           y_spacing=self.offset, modules=modules)
        view = v_win.default_traits_view()
        view.kind = 'live'
        v_win.edit_traits(view=view)
        return v_win
        
    def _b_fired(self):
        self.launch()
        
    def default_traits_view(self):
        v = View(
            VGroup(
                Label('Headstage'),
                UItem('headstage'),
                Tabbed(
                    UItem('file_data', style='custom'),
                    UItem('filters', style='custom'),
                    UItem('module_set',
                          editor=SetEditor(
                              name='all_modules',
                              left_column_title='Analysis modules',
                              right_column_title='Modules to load'
                              )
                        ),
                    ),
                HGroup(
                    VGroup(
                        Label('Offset per channel (in mV)'),
                        UItem('offset', )
                    ),
                    VGroup(
                        Label('Screen Channels?'),
                        UItem('screen_channels')
                    )
                ),
                UItem('b'),
            ),
            resizable=True,
            title='Launch Visualization',
            handler=HeadstageHandler
        )
        return v

    
if __name__ == '__main__':

    ## fd = FileData(
    ##     file='/Users/mike/experiment_data/Viventi 2017-03-27 P4 Rat 17009 Implant/2017-03-27_13-24-53_008_Fs1000.h5',
    ##     data_field='chdata',
    ##     fs_field='Fs',
    ##     chan_map='psv_61_intan2',
    ##     y_scale=0.000198
    ##     )

    ## v = VisLauncher(headstage='intan')
    ## v.configure_traits()
    ## v.file_data = fd

    v = VisLauncher(headstage='mux7')
    v.configure_traits()                    
