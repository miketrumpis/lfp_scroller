from __future__ import division
import os
from glob import glob
from functools import partial
from tempfile import NamedTemporaryFile
import numpy as np
from scipy.signal import detrend
import h5py
from tqdm import tqdm

from traits.api import Instance, Button, HasTraits, File, Float, \
     Str, Property, List, Enum, Int, Bool, on_trait_change, Directory, \
     cached_property
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     Label, Handler, EnumEditor, Tabbed, spring, \
     FileEditor, SetEditor

from ecogana.devices.load.open_ephys import hdf5_open_ephys_channels
from ecogana.devices.load.mux import mux_sampling
from ecoglib.filt.time import downsample
from ecoglib.channel_map import ChannelMap

from .h5data import FilteredReadCache, h5mean, ReadCache, \
     DCOffsetReadCache, CommonReferenceReadCache, interpolate_blanked, H5Chunks
from .filtering import FilterPipeline, pipeline_factory
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

    def create_downsampled(self, where='.'):
        if not os.path.exists(self.file):
            return
        fname = os.path.split(self.file)[1]
        fname, ext = os.path.splitext(fname)
        #ds_fname = os.path.join( os.getcwd(), fname + '_ds.h5' )
        with h5py.File(self.file, 'r') as h5, \
          NamedTemporaryFile(mode='ab', dir=where,
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
                arr_row = arr[i]
                m = np.isnan(arr_row)
                if m.any():
                    arr_row = interpolate_blanked(arr_row, m, inplace=True)
                x_ds, fs = downsample(arr_row, Fs, r=self.ds_rate)
                y[i,:] = x_ds
        return f.name
    
    def _downsample_fired(self):
        self.file = self.create_downsampled()
    
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

from ecogana.devices.load.active_electrodes import get_daq_unmix, pitch_lookup
class ActiveArrayFileData(FileData):

    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    daq = Enum('2t-as daq v1', ('2t-as daq v1', '2t-as daq v2') )
    headstage = Enum('zif26 to 2x uhdmi',
                     ('zif26 to 2x uhdmi',
                      'zif26 to 2x 20 pins harwin to 2x uhdmi',
                      'zif to 50mil',
                      'zif51_p4-50_demux-14c20r'))
    electrode = Enum('actv_64', ('actv_64',
                                 'cardiac v1',
                                 'active_1008ch_sp_v2'))
    y_scale = Property(fget=lambda self: 1e3 / self.gain)
    is_transpose = Property(depends_on='file')
    gain = Float(10)

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return float(f[self.fs_field].value)
        except:
            return None

    def _get_data_channels(self):
        # just returns the contiguous block of channels IN WHICH
        # we find the data carrying channel
        with h5py.File(self.file, 'r') as f:
            ncol = f['numCol'].value
            nrow = f['numRow'].value
        return range(int(ncol * nrow))


    def _get_is_transpose(self):
        if not self.file:
            return
        with h5py.File(self.file, 'r') as fr:
            nrow = int(fr['numRow'].value)
            ncol = int(fr['numChan'].value)
            shape = fr['data'].shape
        return shape[1] == nrow * ncol


    def create_transposed(self, where='.'):
        if not os.path.exists(self.file):
            return
        if not self.is_transpose:
            return self.file
        fname = os.path.split(self.file)[1]
        fname, ext = os.path.splitext(fname)
        # ds_fname = os.path.join( os.getcwd(), fname + '_ds.h5' )
        with h5py.File(self.file, 'r') as h5, \
                NamedTemporaryFile(mode='ab', dir=where,
                                   suffix=fname + '.h5', delete=False) as f:

            arr = h5[self.data_field]
            n_samp, n_chan = arr.shape
            f.file.close()
            fw = h5py.File(f.name, 'w')
            y = fw.create_dataset(self.data_field, shape=(n_chan, n_samp),
                                  dtype=arr.dtype, chunks=True)
            # copy over other keys
            for k in h5.keys():
                if k not in (self.data_field,):
                    try:
                        fw[k] = h5[k].value
                    except (AttributeError, RuntimeError) as e:
                        print 'skipped key', k
            chunk_iter = H5Chunks(h5['data'], axis=0, slices=True)
            #for i in tqdm(xrange(n_chan), desc='Copying transpose'):
            for sl in tqdm(chunk_iter, desc='Copying transpose'):
                arr_rows = arr[sl]
                # m = np.isnan(arr_row)
                # if m.any():
                #     arr_row = interpolate_blanked(arr_row, m, inplace=True)
                y[sl[::-1]] = arr_rows.T
        self.file = f.name
        return f.name


    def make_channel_map(self):
        unmix = get_daq_unmix(self.daq, self.headstage, self.electrode)
        with h5py.File(self.file, 'r') as f:
            #ncol_full = f['numChan'].value
            #ncol_data = f['numCol'].value
            nrow = int(f['numRow'].value)
                          
        pitch = pitch_lookup.get(self.electrode, 1.0)
        # go through channels,
        # if channel is data, put down the array matrix location
        # else, put down a disconnected channel
        data_rows = list(unmix.row)
        data_cols = list(unmix.col)
        chan_map = []
        no_con = []
        for c in self.data_channels:
            col = c // nrow
            row = c % nrow
            if col in data_cols:
                arow = data_rows.index(row)
                acol = data_cols.index(col)
                chan_map.append( arow * len(data_cols) + acol )
            else:
                no_con.append(c)
        nr = len(unmix.row)
        nc = len(unmix.col)
        cm = ChannelMap(chan_map, (nr, nc), pitch=pitch, col_major=False)
        return cm, no_con

    def default_traits_view(self):
        v = super(ActiveArrayFileData, self).default_traits_view()
        vgrp = v.content.content[0]
        new_group = HGroup(
            Item('daq', label='daq'),
            Item('headstage', label='headstage'),
            Item('electrode', label='electrode')
            )
        vgrp.content.insert(0, new_group)
        return v

class BlackrockFileData(FileData):
    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    # 2**15 is 8 mV
    y_scale = Float( 8.0 / 2**15 )

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return f[self.fs_field].value.squeeze()
        except:
            return None


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
                n_row = int(f['numRow'].value)
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

class FilesHandler(Handler):

    def object_file_dir_changed(self, info):
        d = info.object.file_dir
        info.object.joined_files = []
        if not d:
            return
        h5_files = sorted( glob( os.path.join(d, '*.h5') ) )
        h5_files = [os.path.split(h)[1] for h in h5_files]
        h5_files = [os.path.splitext(h)[0] for h in h5_files]
        info.object._available_hdf5_files = h5_files

def concatenate_hdf5_files(files, new_file, filters=None):
    # files is a list of FileData objects
    # filters is possibly a filter pipeline

    # first pass -- get total size and check for consistency
    sizes = []
    samp_res = []
    for fd in files:
        with h5py.File(fd.file, 'r') as h5:
            samp_res.append( h5[fd.fs_field].value )
            sizes.append( h5[fd.data_field].shape )
    samp_res = np.array(samp_res)
    assert (samp_res[0] == samp_res[1:]).all(), 'Inconsistent sampling'
    assert all( [len(s) == 2 for s in sizes] ), 'Only 2D arrays allowed'
    n_chan = sizes[0][0]
    assert all( [s[0] == n_chan for s in sizes] ), 'Inconsistent channels'

    if filters is not None:
        filters = filters.make_pipeline(samp_res[0])
    else:
        filters = pipeline_factory([], None)

    time_size = np.sum( [s[1] for s in sizes] )
    offset = 0
    with h5py.File(new_file, 'w') as fw:
        cat_array = fw.create_dataset( files[0].data_field,
                                       shape=(sizes[0][0], time_size),
                                       dtype='d', chunks=True )
        fw[files[0].fs_field] = samp_res[0]
        for fd in files:
            with h5py.File(fd.file, 'r') as fr:
                array = filters(fr[fd.data_field])
                n = array.shape[1]
                cat_array[:, offset:offset+n] = array[:]
            offset += n
        # copy anything other singleton values from the last file?
        with h5py.File(fd.file, 'r') as fr:
            for k in set(fr.keys()) - {fd.fs_field, fd.data_field}:
                if hasattr(fr[k], 'shape') and not len(fr[k].shape):
                    print 'picked up key', k
                    fw[k] = fr[k].value
    return new_file
        
class ConcatFilesTool(HasTraits):
    file_dir = Directory
    _available_hdf5_files = List(Str)
    joined_files = List([])
    join = Button('Join files')
    downsample = Button('Downsample')
    ds_rate = Int(10)
    filters = Instance(FilterPipeline, ())
    headstage_flavor = Enum(Mux7FileData, (Mux7FileData, OpenEphysFileData))

    file_size = Property(Str, depends_on='joined_files, downsample')
    file_suffix = Str('')
    set_Fs = Property(Str, depends_on='joined_files, downsample')

    @on_trait_change('joined_files')
    def _foo(self):
        print self.joined_files

    def _default_file_map(self):
        if not hasattr(self, '_file_map'):
            self._file_map = dict()
        klass = self.headstage_flavor
        # Rules:
        # 1) keep any existing file name to FileData map
        # 2) create new FileData map for novel keys
        # 3) remove unused/stale keys
        for f in self.joined_files:
            if f in self._file_map:
                continue
            full_file = os.path.join(self.file_dir, f + '.h5')
            fd = klass(file=full_file)
            self._file_map[f] = fd
        for f in ( set(self._file_map.keys()) - set(self.joined_files) ):
            self._file_map.pop(f)
        return self._file_map
    
    def _join_fired(self):
        if not len(self.joined_files):
            return
        self._default_file_map()
        file_names = self.joined_files
        file_objs = [self._file_map[f] for f in file_names]
        new_file_name = '-'.join(file_names)
        if self.file_suffix:
            new_file_name = new_file_name + '_' + self.file_suffix
        full_file = os.path.join(self.file_dir, new_file_name + '.h5')
        concatenate_hdf5_files(file_objs, full_file, filters=self.filters)

    def _downsample_fired(self):
        if not len(self.joined_files):
            return
        klass = self.headstage_flavor
        self._default_file_map()
        for f in self.joined_files:
            # use existing FileData if already mapped
            # (e.g. for sequential resampling)
            if f in self._file_map:
                fd = self._file_map[f]
            else:
                full_file = os.path.join(self.file_dir, f + '.h5')
                fd = klass(file=full_file, ds_rate=self.ds_rate)
            fd.file = fd.create_downsampled(where=self.file_dir)
            self._file_map[f] = fd
        self.file_size

    @cached_property
    def _get_set_Fs(self):
        if not len(self.joined_files):
            return ''
        samps = np.array([fd.Fs for fd in self._default_file_map().values()])
        if (samps[1:] == samps[0]).all():
            return '{0:.1f}'.format(samps[0])
        else:
            return 'Mixed!'

    @cached_property
    def _get_file_size(self):
        fs = 0
        for fd in self._default_file_map().values():
            fs += os.stat( fd.file ).st_size
        prefix = ('', 'k', 'M', 'G', 'T')
        if fs > 0:
            size = np.log2(fs)
            n = int(size // 10)
            size = 2 ** ( size - n * 10.0 )
        else:
            size = 0
            n = 0
        order = prefix[n] + 'b'
        return '{0:.2f} {1}'.format(size, order)
        
    def default_traits_view(self):
        v = View(
                VGroup(
                    Item('headstage_flavor', label='HS Type'),
                    Item('file_dir', label='Data directory'),
                    Tabbed(
                        UItem('joined_files',
                              editor=SetEditor(
                                  name='_available_hdf5_files',
                                  left_column_title='Available files',
                                  right_column_title='Files to concat',
                                  ordered=True
                                  )
                            ),
                        UItem('filters', style='custom')
                    ),
                    HGroup(
                        UItem('join'),
                        spring,
                        Item('file_suffix', label='File suffix'),
                        spring,
                        Item('file_size', label='Estimated file size',
                             style='readonly')
                        ),
                    HGroup(
                        UItem('downsample'),
                        Item('ds_rate', label='Downsample factor'),
                        Item('set_Fs', label='Fs for set',
                             style='readonly')

                        )
                    ),
                handler=FilesHandler,
                title='File concatenation tool',  
                resizable=True
                )
        return v

