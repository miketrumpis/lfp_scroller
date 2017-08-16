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
from tqdm import tqdm

from traits.api import Instance, Button, HasTraits, File, Float, \
     Str, Property, List, Enum, Int, Bool, on_trait_change
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     Label, Handler, EnumEditor, Group, TableEditor, Tabbed, spring, \
     FileEditor, SetEditor
from traitsui.table_column import ObjectColumn

from ecogana.devices.electrode_pinouts import get_electrode_map, electrode_maps
from ecogana.devices.load.open_ephys import hdf5_open_ephys_channels
from ecogana.devices.load.mux import mux_sampling
from ecogana.devices.channel_picker import interactive_mask
from ecoglib.filt.time.design import butter_bp, cheby1_bp, cheby2_bp, \
     notch, ellip_bp, savgol
from ecoglib.filt.time import ma_highpass, downsample
from ecoglib.util import ChannelMap, Bunch

from .h5scroller import FastScroller
from .h5data import bfilter, FilteredReadCache, h5mean, ReadCache, \
     DCOffsetReadCache, CommonReferenceReadCache

from .new_scroller import VisWrapper
from .modules import ana_modules, default_modules
from . import Error

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
    zero_windows = Bool
    car_windows = Bool
    data_channels = Property
    Fs = Property(depends_on='file')
    downsample = Button('Downsample')
    ds_rate = Int(10)

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return f[self.fs_field].value
        except:
            return None

    def _get_data_channels(self):
        if not os.path.exists(self.file):
            return []
        try:
            with h5py.File(self.file, 'r') as f:
                if self.data_field in f:
                    return range( f[self.data_field].shape[0] )
                else:
                    return []
        except IOError:
            return []
        
    def _compose_arrays(self, filter):
        # default is just a wrapped array
        h5 = h5py.File(self.file, 'r')
        array = filter(h5[self.data_field])
        if self.zero_windows:
            array = FilteredReadCache(array, partial(detrend, type='constant'))
        elif self.car_windows:
            array = CommonReferenceReadCache(array)
        else:
            array = ReadCache(array)
        return array

    def _downsample_fired(self):
        if not os.path.exists(self.file):
            return
        fname = os.path.split(self.file)[1]
        fname, ext = os.path.splitext(fname)
        #ds_fname = os.path.join( os.getcwd(), fname + '_ds.h5' )
        with h5py.File(self.file, 'r') as h5, \
          NamedTemporaryFile(mode='ab', dir='.',
                             suffix=fname+'.h5', delete=False) as f:
        
            arr = h5[self.data_field]
            Fs = h5[self.fs_field].value
            n_chan, len_ts = arr.shape
            n = len_ts // self.ds_rate
            if n * self.ds_rate < len_ts:
                n += 1
            f.file.close()
            fw = h5py.File(f.name, 'w')
            y = fw.create_dataset(self.data_field, shape=(n_chan, n),
                                  dtype=arr.dtype, chunks=True)
            fw[self.fs_field] = Fs / float(self.ds_rate)
            # copy over other keys
            for k in h5.keys():
                if k not in (self.data_field, self.fs_field):
                    try:
                        fw[k] = h5[k].value
                    except (AttributeError, RuntimeError) as e:
                        print 'skipped key', k
                    
            for i in tqdm(xrange(n_chan), desc='Downsampling'):
                x_ds, fs = downsample(arr[i], Fs, r=self.ds_rate)
                y[i,:] = x_ds
        self.file = f.name
    
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
                HGroup(
                    Item('zero_windows', label='Remove local DC?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    ),
                HGroup(
                    UItem('downsample'),
                    Item('ds_rate', label='Downsample rate'),
                    Item('Fs', label='Current Fs', style='readonly')
                    )
                ),
            handler=FileHandler,
            resizable=True
        )
        return view

class OpenEphysFileData(FileData):
    file = File(filter=[u'*.h5', u'*.continuous'])
    data_field = Property(fget=lambda self: 'chdata')
    fs_field = Property(fget=lambda self: 'Fs')
    y_scale = Float(0.001)

    # convert controls
    can_convert = Property
    temp_conv = Bool(False)
    conv_file = File
    downsamp = Int(1)
    quantized = Bool(False)
    do_convert = Button('Run convert')
    
    def _get_can_convert(self):
        if not self.file or not os.path.exists(self.file):
            return False
        return os.path.splitext(self.file)[1] == '.continuous'
    
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
                        Item('downsamp', label='Downsample factor', width=4),
                        Item('quantized', label='Quantized?'),
                        label='Conversion options'
                        ),
                    UItem('do_convert'),
                    enabled_when='can_convert'
                    ),
                    
                Label('Scales samples to mV'),
                Label('(1.98e-4 if quantized, else 1e-3)'),
                UItem('y_scale'),
                HGroup(
                    Item('zero_windows', label='Remove local DC?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    )
                ),
            resizable=True
        )
        return v
        
_sampling = mux_sampling.keys() + ['mux3', 'unknown']
class Mux7FileData(FileData):
    """
    An HDF5 array holding timeseries from the MUX headstages:
    defaults to building an ReadCache with DC offset subraction.
    """

    y_scale = Property(fget=lambda self: 1e3 / self.gain)
    gain = Enum( 12, (3, 4, 10, 12, 20) )
    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    dc_subtract = Bool(True)
    sampling_style = Enum( 'mux7_1card', _sampling )

    def _get_data_channels(self):
        if not os.path.exists(self.file):
            return []
        try:
            with h5py.File(self.file, 'r') as f:
                if not self.data_field in f:
                    return []
                n_chan = f[self.data_field].shape[0]
                n_row = f['numRow'].value
                n_col = f['numCol'].value
                dig_order = mux_sampling.get(self.sampling_style, range(4))
                chans  = [ range(i*n_row, (i+1)*n_row) for i in dig_order ]
                return reduce(lambda x, y: x + y, chans)
        except IOError:
            return []
    
    def _compose_arrays(self, filter):
        if not self.dc_subtract or self.zero_windows or self.car_windows:
            print 'not removing dc'
            return super(Mux7FileData, self)._compose_arrays(filter)
        f = h5py.File(self.file, 'r')
        array = filter(f[self.data_field])
        mn_level = h5mean(array, 1, start=int(3e4))
        array = DCOffsetReadCache(array, mn_level)
        return array

    def default_traits_view(self):
        view = View(
            VGroup(
                Item('file', label='Visualize this file'),
                HGroup(
                    VGroup(
                        Label('Amplifier gain (default 12 for Mux v7)'),
                        UItem('gain')
                        ),
                    Item('dc_subtract', label='DC compensation'),
                    Item('sampling_style', label='DAQ setup')
                    ),
                HGroup(
                    Item('zero_windows', label='Remove DC per window?'),
                    Item('car_windows', label='Com. Avg. Ref?'),
                    label='Choose up to one'
                    ),
                HGroup(
                    UItem('downsample'),
                    Item('ds_rate', label='Downsample rate'),
                    Item('Fs', label='Current Fs', style='readonly')
                    )
                ),
            resizable=True
            )
        return view

class FilterMenu(HasTraits):
    """Edits filter arguments."""
    filtfilt = Bool(False, info_text='Forward/reverse filter')
    def default_traits_view(self):
        # builds a panel automatically from traits
        t = self.traits()
        items = list()
        for k, v in t.items():
            if k in ('trait_added', 'trait_modified'):
                continue
            if v.info_text:
                items.extend( [Label(v.info_text), UItem(k, style='simple')] )
            else:
                items.extend( [UItem(k, style='simple')] )
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
            self.stop_atten, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
            )
    
class NotchMenu(FilterMenu):
    notch = Float(60, info_text='notch frequency')
    nwid = Float(3.0, info_text='notch width (Hz)')
    nzo = Int(3, info_text='number of zeros at notch')
    def make_filter(self, Fs):
        return notch(self.notch, Fs, self.nwid, nzo=self.nzo)

class MAHPMenu(FilterMenu):
    hpc = Float(1, info_text='highpass cutoff')
    def make_filter(self, Fs):
        fir = ma_highpass(None, self.hpc/Fs, fir_filt=True)
        return fir, 1

class EllipMenu(BandpassMenu):
    atten = Float(10, info_text='Stopband attenutation (dB)')
    ripple = Float(0.5, info_text='Passband ripple (dB)')
    hp_width = Float(0, info_text='HP transition width (Hz)')
    lp_width = Float(0, info_text='LP transition width (Hz)')
    ord = Int(info_text='filter order')
    samp_rate = Float(2.0, info_text='Sampling rate')
    check_order = Button('check order')

    def _check_order_fired(self):
        b, a = self.make_filter(self.samp_rate)
        self.ord = len(b)

    def make_filter(self, Fs):
        return ellip_bp(
            self.atten, self.ripple, lo=self.lo, hi=self.hi,
            hp_width=self.hp_width, lp_width=self.lp_width, Fs=Fs
            )

class SavgolMenu(FilterMenu):
    T = Float(1.0, info_text='Window length (s)')
    ord = Int(3, info_text='Poly order')
    sm = Bool(True, info_text='Smoothing (T), residual (F)')

    def make_filter(self, Fs):
        return savgol(self.T, self.ord, Fs=Fs, smoothing=self.sm)
    
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
        elif info.object.filt_type == 'elliptical':
            if not isinstance(info.object.filt_menu, EllipMenu):
                info.object.filt_menu = EllipMenu()            
        elif info.object.filt_type == 'notch':
            if not isinstance(info.object.filt_menu, NotchMenu):
                info.object.filt_menu = NotchMenu()
        elif info.object.filt_type == 'm-avg hp':
            if not isinstance(info.object.filt_menu, MAHPMenu):
                info.object.filt_menu = MAHPMenu()
        elif info.object.filt_type == 'savitzky-golay':
            if not isinstance(info.object.filt_menu, SavgolMenu):
                info.object.filt_menu = SavgolMenu()
            

available_filters = ('butterworth',
                     'cheby 1',
                     'cheby 2',
                     'elliptical',
                     'notch',
                     'm-avg hp',
                     'savitzky-golay')

class Filter(HasTraits):
    """Filter model with adaptive menu. Menu can build transfer functions."""
    
    filt_type = Enum( 'butterworth', available_filters )
    filt_menu = Instance(HasTraits)

    view = View(
        VGroup(
            Item('filt_type', label='Filter type'),
            Group(
                UItem('filt_menu', style='custom'),
                label='Filter menu'
                )
            ),
        handler=FilterHandler,
        resizable=True
        )
    
filter_table = TableEditor(
    columns = [ ObjectColumn( name='filt_type' ) ],
    deletable=True,
    auto_size=True,
    show_toolbar=True,
    reorderable=True,
    edit_view=None,
    row_factory=Filter,
    selected='selected'
    )

def _pipeline_factory(f_list, filter_modes):
    if not len(f_list):
        return lambda x: x
    def copy_x(x):
        with NamedTemporaryFile(mode='ab', dir='.') as f:
            f.file.close()
            fw = h5py.File(f.name, 'w')
            y = fw.create_dataset('data', shape=x.shape,
                                  dtype=x.dtype, chunks=x.chunks)
        return y
        
    def run_list(x, axis=1):
        y = copy_x(x)
        b, a = f_list[0]
        bfilter(b, a, x, axis=axis, out=y, filtfilt=filter_modes[0])
        for (b, a), ff in zip(f_list[1:], filter_modes[1:]):
            bfilter(b, a, y, axis=axis, filtfilt=ff)
        return y

    return run_list
        

class FilterPipeline(HasTraits):
    """List of filters that can be pipelined."""
    filters = List(Filter)
    filt_type = Enum( 'butterworth', available_filters )
    selected = Instance(Filter)
    add_filter = Button('Add filter')
    remove_filter = Button('Remove filter')

    def make_pipeline(self, Fs):
        filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
        filtfilt = [f.filt_menu.filtfilt for f in self.filters]
        return _pipeline_factory(filters, filtfilt)

    def _add_filter_fired(self):
        self.filters.append( Filter(filt_type=self.filt_type) )

    def _remove_filter_fired(self):
        self.filters.remove( self.selected )
    
    view = View(
        VGroup(
            Group(
                UItem('filt_type'),
                UItem('add_filter'),
                UItem('remove_filter'),
                springy=True
                ),
            Group(
                UItem('filters', editor=filter_table),
                show_border=True
                ),
            ),
        resizable=True,
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
        elif hs.lower() == 'stim v4':
            fd = Mux7FileData(gain=4)
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
    offset = Enum(0.2, [0, 0.1, 0.2, 0.5, 1, 2, 5])
    max_window_width = Float(1200.0)
    headstage = Enum('mux7',
                     ('mux3', 'mux5', 'mux6', 'mux7',
                      'stim v4', 'intan', 'unknown'))
    chan_map = Enum(sorted(electrode_maps.keys())[0],
                    sorted(electrode_maps.keys()) + ['unknown'])
    n_chan = Int
    skip_chan = Str
    elec_geometry = Str
    screen_channels = Bool(False)
    screen_start = Float(0)

    def __init__(self, **traits):
        super(VisLauncher, self).__init__(**traits)
        self.add_trait('filters', FilterPipeline())

    def _get_screen(self, array, channels, chan_map, Fs):
        from ecogana.expconfig import params
        mem_guideline = float(params.memory_limit)
        n_chan = len(array)
        word_size = array.dtype.itemsize
        n_pts = min(1000000, mem_guideline / n_chan / word_size)
        offset = int(self.screen_start * 60 * Fs)
        n_pts = min(array.shape[1]-offset, n_pts)
        data = np.empty( (len(channels), n_pts), dtype=array.dtype )
        for n, c in enumerate(channels):
            data[n] = array[c, offset:offset+n_pts]

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
        if self.chan_map == 'unknown':
            chan = np.arange(self.n_chan)
            try:
                nc = np.array( map(int, self.skip_chan.split(',')) )
            except:
                nc = []
            geo = map(int, self.elec_geometry.split(','))
            n_sig_chan = self.n_chan - len(nc)
            chan_map = ChannelMap(np.arange(n_sig_chan), geo)
        else:
            chan_map, nc = get_electrode_map(self.chan_map)

        with h5py.File(self.file_data.file, 'r') as h5:
            x_scale = h5[self.file_data.fs_field].value ** -1.0
            array_size = h5[self.file_data.data_field].shape[0]
        num_vectors = len(chan_map) + len(nc)
        
        data_channels = [self.file_data.data_channels[i]
                         for i in xrange(num_vectors) if i not in nc]

        # permute  channels to stack rows
        chan_idx = zip(*chan_map.to_mat())
        chan_order = chan_map.lookup(*zip( *sorted(chan_idx)[::-1] ))
        data_channels = [data_channels[i] for i in chan_order]
        chan_map = ChannelMap( [chan_map[i] for i in chan_order],
                               chan_map.geometry, col_major=chan_map.col_major )

        filters = self.filters.make_pipeline(x_scale ** -1.0)
        array = self.file_data._compose_arrays(filters)
        if self.screen_channels:
            data_channels, chan_map = \
              self._get_screen(array, data_channels, chan_map, x_scale**-1.0)
            
        rm = np.zeros( (array_size,), dtype='?' )
        rm[data_channels] = True
              

        nav = h5mean(array.file_array, 0, rowmask=rm)
        nav *= self.file_data.y_scale

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
                HGroup(
                    VGroup(
                        Label('Headstage'),
                        UItem('headstage'),
                        ),
                    VGroup(
                        Label('Channel map'),
                        UItem('chan_map')
                        ),
                    HGroup(
                        VGroup(
                            Label('N signal channelsl'),
                            UItem('n_chan'),
                            ),
                        VGroup(
                            Label('Skip chans'),
                            UItem('skip_chan')
                            ),
                        VGroup(
                            Label('Grid geometry'),
                            UItem('elec_geometry')
                            ),
                        visible_when='chan_map=="unknown"'
                        ),
                    ),
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
                    ),
                    VGroup(
                        Label('Begin screening at x minutes'),
                        UItem('screen_start')
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

    v = VisLauncher(headstage='mux7', chan_map='ratv4_mux6')
    v.configure_traits()                    
