import os
from itertools import product
from pathlib import Path
from glob import glob
from functools import partial
from contextlib import ExitStack, contextmanager
from tempfile import NamedTemporaryFile
import numpy as np
from scipy.signal import detrend
import h5py
from tqdm import tqdm
import json
from spikeinterface.extractors import OpenEphysLegacyRecordingExtractor
from pynwb import NWBHDF5IO
from pynwb.ecephys import ElectricalSeries


from traits.api import Instance, Button, HasTraits, File, Float, \
     Str, Property, List, Enum, Int, Bool, on_trait_change, Directory, \
     cached_property
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     Label, Handler, EnumEditor, Tabbed, spring, \
     FileEditor, SetEditor
from pyface.api import DirectoryDialog, MessageDialog, OK

from ecogdata.devices.load.open_ephys import hdf5_open_ephys_channels
from ecogdata.devices.load.mux import mux_sampling
from ecogdata.devices.load.active_electrodes import get_daq_unmix, pitch_lookup
from ecogdata.filt.time import downsample
from ecogdata.channel_map import ChannelMap

from .h5data import FilteredReadCache, h5mean, ReadCache, OpenEphysExtractorWrapper, NwbWrapper, \
     DCOffsetReadCache, CommonReferenceReadCache, interpolate_blanked, H5Chunks, passthrough
from .filtering import FilterPipeline, PipelineFactory
from .helpers import Error, PersistentWindow


class FileHandler(Handler):

    fields = List(Str)
    def object_file_changed(self, info):
        if not len(info.object.file):
            self.fields = []
            return
        h5 = h5py.File(info.object.file, 'r')
        self.fields = list(h5.keys())
        h5.close()


class NWBFileHandler(Handler):

    fields = List(Str)
    def object_file_changed(self, info):
        if not len(info.object.file):
            self.fields = []
            return
        with NWBHDF5IO(info.object.file, 'r') as io:
            nwbfile = io.read()
            es_list = [i.name for i in nwbfile.all_children() if isinstance(i, ElectricalSeries)]
        self.fields = es_list
        if info.object.data_field:
            self.object_data_field_changed(info)

    def object_data_field_changed(self, info):
        if not len(info.object.file):
            return
        with info.object._get_eseries(info.object.data_field, close_io=True) as es:
            if es.rate is not None:
                info.object.Fs = es.rate
            else:
                deltas = np.diff(es.timestamps[:2000])
                info.object.Fs = 1 / deltas.mean()
            info.object.y_scale = es.conversion


class FileData(HasTraits):
    """Basic model for an HDF5 file holding an array timeseries."""
    # The "data field" of the hdf5 file should be a floating point array (which may be violated by the black rock
    # subclass). Downsampling and filtering do not re-scale the voltage values, so intermediate files have the same
    # gain/scale as the input files.
    file = File(filter=['*.h5'], exists=True)
    data_field = Str
    fs_field = Str
    y_scale = Float(1.0)
    zero_windows = Bool
    car_windows = Bool
    data_channels = Property
    Fs = Property(depends_on='file')
    downsample = Button('Downsample')
    ds_rate = Int(10)
    array_size = Property(depends_on='file,data_field')

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return f[self.fs_field][()]
        except:
            return None

    def _get_data_channels(self):
        shape = self.array_size
        if not shape:
            return list()
        return list(range(shape[0]))

    def _get_array_size(self):
        if not os.path.exists(self.file):
            return ()
        try:
            with h5py.File(self.file, 'r') as f:
                if self.data_field in f:
                    return f[self.data_field].shape
                else:
                    return ()
        except IOError:
            return ()

    def compose_arrays(self, filter: PipelineFactory):
        # default is just a wrapped array
        h5 = h5py.File(self.file, 'r')
        array = filter.filter_pipeline(h5[self.data_field])
        timestamps = np.arange(array.shape[1]) / self.Fs
        if self.zero_windows:
            array = FilteredReadCache(array, partial(detrend, type='constant'))
        elif self.car_windows:
            array = CommonReferenceReadCache(array)
        else:
            array = ReadCache(array)
        return array, timestamps

    def create_downsampled(self, where='.'):
        # This does not convert units! units out == units in
        if not os.path.exists(self.file):
            return
        fname = os.path.split(self.file)[1]
        fname, ext = os.path.splitext(fname)
        ds_name = os.path.join(where, fname + '_dnsamp{}.h5'.format(self.ds_rate))
        Fs = self.Fs
        with h5py.File(self.file, 'r') as h5, h5py.File(ds_name, 'w') as fw:
            arr = h5[self.data_field]
            n_chan, len_ts = arr.shape
            n = len_ts // self.ds_rate
            if n * self.ds_rate < len_ts:
                n += 1
            y = fw.create_dataset(self.data_field, shape=(n_chan, n),
                                  dtype=arr.dtype, chunks=True)
            fw[self.fs_field] = Fs / float(self.ds_rate)
            # copy over other keys
            for k in h5.keys():
                if k not in (self.data_field, self.fs_field):
                    try:
                        fw[k] = h5[k][()]
                    except (AttributeError, RuntimeError) as e:
                        print('skipped key:', k)
                    
            for i in tqdm(range(n_chan), desc='Downsampling'):
                arr_row = arr[i]
                m = np.isnan(arr_row)
                if m.any():
                    arr_row = interpolate_blanked(arr_row, m, inplace=True)
                x_ds, fs = downsample(arr_row, Fs, r=self.ds_rate)
                y[i,:] = x_ds
        return Path(ds_name)
    
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
                Label('Scales samples to voltage (V)'),
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


class NWBFileData(FileData):
    file = File(filter=['*.nwb'], exists=True)
    data_field = Str
    Fs = Float(1.0)
    map_nonsignal_channels = Bool(False)

    @contextmanager
    def _get_eseries(self, electrical_series_name: str, close_io: bool=True):
        with ExitStack() as stack:
            if close_io:
                io = stack.enter_context(NWBHDF5IO(self.file, 'r'))
            else:
                io = NWBHDF5IO(self.file, 'r')
            nwbfile = io.read()
            es_dict = {i.name: i for i in nwbfile.all_children() if isinstance(i, ElectricalSeries)}
            assert electrical_series_name in es_dict, 'electrical series name not present in nwbfile'
            es = es_dict[electrical_series_name]
            yield es
        # return es

    def _get_data_channels(self):
        if hasattr(self, '_cached_data_channels'):
            return self._cached_data_channels
        with self._get_eseries(self.data_field, close_io=True) as series:
            electrodes = series.electrodes.to_dataframe()
        self._cached_data_channels = list(range(len(electrodes)))
        # Correction -- this should return all the channels in the timeseries as "data_channels".
        # It is up to the "make_channel_map()" method to decide which channels are mapped or not
        return self._cached_data_channels

    def _get_array_size(self):
        if not os.path.exists(self.file):
            return ()
        with self._get_eseries(self.data_field, close_io=True) as series:
            shape = series.data.shape
        return shape[::-1]

    def make_channel_map(self):
        with self._get_eseries(self.data_field, close_io=True) as series:
            electrodes = series.electrodes.to_dataframe()
        row = electrodes.row.values
        col = electrodes.col.values
        ref_type = np.isnan(row)
        sig_row = row[~ref_type].astype('i')
        sig_col = col[~ref_type].astype('i')
        if self.map_nonsignal_channels:
            grid_values = set(product(range(sig_row.max()), range(sig_col.max())))
            unused_grid_iter = iter(sorted(list(grid_values - set(zip(sig_row, sig_col)))))
            for ref_idx in np.where(ref_type)[0]:
                fake_row, fake_col = next(unused_grid_iter)
                print('ref channel', electrodes.index[ref_idx], 'going to', fake_row, fake_col)
                row[ref_idx] = fake_row
                col[ref_idx] = fake_col
            col = col.astype('i')
            row = row.astype('i')
            ref_idx = []
        else:
            row = sig_row
            col = sig_col
            ref_idx = np.where(ref_type)[0]
        cm = ChannelMap.from_index(list(zip(row, col)), None)
        return cm, ref_idx

    def compose_arrays(self, filter: PipelineFactory):
        # default is just a wrapped array
        # h5 = h5py.File(self.file, 'r')
        with self._get_eseries(self.data_field, close_io=False) as series:
            pass
        # array = series.data
        # filter.transpose = True
        array = NwbWrapper(series)
        if series.rate is not None:
            timestamps = np.arange(array.shape[1]) / series.rate
        else:
            timestamps = series.timestamps[:]
        # array = filter.filter_pipeline(array)
        if self.zero_windows:
            array = FilteredReadCache(array, partial(detrend, type='constant'))
        elif self.car_windows:
            array = CommonReferenceReadCache(array)
        else:
            array = ReadCache(array)
        return array, timestamps

    def default_traits_view(self):
        v = super().default_traits_view()
        v.handler = NWBFileHandler
        hgroup = v.content.content[0].content[1]
        # Get rid of the fs_field element (this is determined by the Timeseries object)
        hgroup.content.pop(index=1)
        hgroup.content.append(Item('map_nonsignal_channels', label='Include unmapped channels'))
        # Change y_scale to read-only
        v.content.content[0].content[3] = UItem('y_scale', style='readonly')
        return v


class RHDFileData(FileData):
    y_scale = Float(0.195 * 1e-6)  # Default Intan gain
    data_field = Str('amplifier_data')
    fs_field = Property

    def _get_fs_field(self):
        return 'n/a'

    def _get_Fs(self):
        if not self.file:
            return
        try:
            with h5py.File(self.file, 'r') as f:
                header = json.loads(f.attrs['JSON_header'])
            return header['sample_rate']
        except:
            return

    def create_downsampled(self, where='.'):
        ds_name = super().create_downsampled(where=where)
        # now copy the JSON header
        with h5py.File(self.file, 'r') as f:
            header = json.loads(f.attrs['JSON_header'])
        header['sample_rate'] = self.Fs / self.ds_rate
        with h5py.File(ds_name, 'r+') as f:
            f.attrs['JSON_header'] = json.dumps(header)
        return ds_name


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
    y_scale = Property(fget=lambda self: 1.0 / self.gain)
    is_transpose = Property(depends_on='file')
    gain = Float(10)

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return float(f[self.fs_field][()])
        except:
            return None

    def _get_data_channels(self):
        # just returns the contiguous block of channels IN WHICH
        # we find the data carrying channel
        with h5py.File(self.file, 'r') as f:
            ncol = f['numCol'][()]
            nrow = f['numRow'][()]
        return list(range(int(ncol * nrow)))


    def _get_is_transpose(self):
        if not self.file:
            return
        with h5py.File(self.file, 'r') as fr:
            nrow = int(fr['numRow'][()])
            ncol = int(fr['numChan'][()])
            shape = fr['data'].shape
        return shape[1] == nrow * ncol

    def compose_arrays(self, filter: PipelineFactory):
        if self.is_transpose:
            filter.transpose = True
        return super().compose_arrays(filter)

    # def create_transposed(self, where='.'):
    #     if not os.path.exists(self.file):
    #         return
    #     if not self.is_transpose:
    #         return self.file
    #     fname = os.path.split(self.file)[1]
    #     fname, ext = os.path.splitext(fname)
    #     # ds_fname = os.path.join( os.getcwd(), fname + '_ds.h5' )
    #     with h5py.File(self.file, 'r') as h5, \
    #             NamedTemporaryFile(mode='ab', dir=where,
    #                                suffix=fname + '.h5', delete=False) as f:
    #
    #         arr = h5[self.data_field]
    #         n_samp, n_chan = arr.shape
    #         f.file.close()
    #         fw = h5py.File(f.name, 'w')
    #         y = fw.create_dataset(self.data_field, shape=(n_chan, n_samp),
    #                               dtype=arr.dtype, chunks=True)
    #         # copy over other keys
    #         for k in h5.keys():
    #             if k not in (self.data_field,):
    #                 try:
    #                     fw[k] = h5[k][()]
    #                 except (AttributeError, RuntimeError) as e:
    #                     print('skipped key:', k)
    #         # keep chunk size to ~ 200 MB
    #         min_size = int(200e6 / 8 / n_chan)
    #         chunk_iter = H5Chunks(h5['data'], axis=0, slices=True, min_chunk=min_size)
    #         #for i in tqdm(range(n_chan), desc='Copying transpose'):
    #         for sl in tqdm(chunk_iter, desc='Copying transpose'):
    #             arr_rows = arr[sl]
    #             # m = np.isnan(arr_row)
    #             # if m.any():
    #             #     arr_row = interpolate_blanked(arr_row, m, inplace=True)
    #             y[sl[::-1]] = arr_rows.T
    #     self.file = f.name
    #     return f.name

    def make_channel_map(self):
        unmix = get_daq_unmix(self.daq, self.headstage, self.electrode)
        with h5py.File(self.file, 'r') as f:
            #ncol_full = f['numChan'][()]
            #ncol_data = f['numCol'][()]
            nrow = int(f['numRow'][()])
                          
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
                chan_map.append(arow * len(data_cols) + acol)
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
    y_scale = Float( 8e-3 / 2**15 )

    def _get_Fs(self):
        try:
            with h5py.File(self.file, 'r') as f:
                return f[self.fs_field][()].squeeze()
        except:
            return None

# TODO: make a open ephys variance that uses spikeinterface/neo to map the file..
#  then the _compose_array method will just return something that maps __getitem__ to get_traces()
class SIOpenEphysFileData(FileData):
    file = Directory(exists=True)
    y_scale = Float(1.95e-7)
    data_field = Property
    fs_field = Property
    _extractor: OpenEphysLegacyRecordingExtractor=None

    @on_trait_change('file')
    def _update_extractor(self):
        if not self.file:
            self._extractor = None
            return
        if self._extractor is not None:
            old_path = self._extractor.neo_reader.dirname
        else:
            old_path = ''
        if old_path != self.file:
            self._extractor = OpenEphysLegacyRecordingExtractor(self.file, stream_id='CH')

    def _get_data_field(self):
        return 'CH'

    def _get_fs_field(self):
        return ''

    def _get_Fs(self):
        if self._extractor is None:
            return None
        return self._extractor.get_sampling_frequency()

    def _get_data_channels(self):
        if not os.path.exists(self.file):
            return []
        if self._extractor is None:
            return []
        else:
            return list(range(self._extractor.get_num_channels()))

    def _get_array_size(self):
        if self._extractor is None:
            return ()
        return (len(self.data_channels), self._extractor.get_num_samples())

    def compose_arrays(self, filter: PipelineFactory):
        si_wrap = OpenEphysExtractorWrapper(self._extractor)
        array = filter.filter_pipeline(si_wrap)
        # # default is just a wrapped array
        # h5 = h5py.File(self.file, 'r')
        # array = filter(h5[self.data_field])
        timestamps = np.arange(array.shape[1]) / self.Fs
        if self.zero_windows:
            array = FilteredReadCache(array, partial(detrend, type='constant'))
        elif self.car_windows:
            array = CommonReferenceReadCache(array)
        else:
            array = ReadCache(array)
        return array, timestamps

    def default_traits_view(self):
        v = super().default_traits_view()
        # Unlink handler
        v.handler = None
        vg = v.content.content[0]
        # get rid of data and samp rate field info
        vg.content.pop(1)
        # get rid of the downsample stuff in the parent view
        final_group = vg.content[-1]
        new_list = list()
        for item in final_group.content:
            if item.name in ('downsample', 'ds_rate'):
                continue
            new_list.append(item)
        final_group.content = new_list
        return v


class OpenEphysFileData(FileData):
    file = File(filter=['*.h5', '*.continuous'], exists=True)
    data_field = Property(fget=lambda self: 'chdata')
    fs_field = Property(fget=lambda self: 'Fs')
    y_scale = Float(1e-6)

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
            print('already an HDF5 file!!')
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
            print('converting to', self.conv_file)
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
                    
                Label('Scales samples to V'),
                Label('(1.98e-7 if quantized, else 1e-6)'),
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


_sampling = list(mux_sampling.keys()) + ['mux3', 'unknown']
class Mux7FileData(FileData):
    """
    An HDF5 array holding timeseries from the MUX headstages:
    defaults to building an ReadCache with DC offset subraction.
    """

    y_scale = Property(fget=lambda self: 1.0 / self.gain)
    gain = Enum(12, (3, 4, 10, 12, 20))
    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    dc_subtract = Bool(False)
    sampling_style = Enum('mux7_1card', _sampling)

    def _get_data_channels(self):
        if not os.path.exists(self.file):
            return []
        try:
            with h5py.File(self.file, 'r') as f:
                if not self.data_field in f:
                    return []
                n_row = int(f['numRow'][()])
                dig_order = mux_sampling.get(self.sampling_style, list(range(4)))
                chans = list()
                for i in dig_order:
                    chans.extend(range(i*n_row, (i+1)*n_row))
            return chans
        except IOError:
            return []
    
    def compose_arrays(self, filter: PipelineFactory):
        if not self.dc_subtract or self.zero_windows or self.car_windows:
            print('not removing dc')
            return super(Mux7FileData, self).compose_arrays(filter)
        f = h5py.File(self.file, 'r')
        array = filter.filter_pipeline(f[self.data_field])
        mn_level = h5mean(array, 1, start=int(3e4))
        array = DCOffsetReadCache(array, mn_level)
        timestamps = np.arange(array.shape[1]) / self.Fs
        return array, timestamps

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
        if os.path.split(d)[1] == 'untitled' and not os.path.exists(d):
            d = os.path.split(d)[0]
            info.object.file_dir = d
        info.object.working_files = []
        if not d:
            return
        h5_files = sorted(glob(os.path.join(d, '*.h5')))
        h5_files = [os.path.split(h)[1] for h in h5_files]
        h5_files = [os.path.splitext(h)[0] for h in h5_files]
        info.object._available_hdf5_files = h5_files


def _check_consistency(files):
    sizes = []
    samp_res = []
    for fd in files:
        with h5py.File(fd.file, 'r') as h5:
            # samp_res.append( h5[fd.fs_field][()] )
            samp_res.append(fd.Fs)
            sizes.append(h5[fd.data_field].shape)
    samp_res = np.array(samp_res)
    assert (samp_res[0] == samp_res[1:]).all(), 'Inconsistent sampling'
    assert all([len(s) == 2 for s in sizes]), 'Only 2D arrays allowed'
    n_chan = sizes[0][0]
    assert all([s[0] == n_chan for s in sizes]), 'Inconsistent channels'


def concatenate_hdf5_files(files, new_file, filters=None):
    # files is a list of FileData objects
    # filters is possibly a filter pipeline

    # first pass -- get total size and check for consistency
    _check_consistency(files)

    # Here, the filtering pipeline will create the output file with the correct size
    samp_res = files[0].Fs
    data_field = files[0].data_field
    if filters is not None and len(filters):
        print('Concatenating with given filters')
        filters = filters.make_pipeline(samp_res, output_file=new_file).filter_pipeline
    else:
        # in thise case, just create a new file using passthrough
        print('Concatenating with passthrough')
        filters = PipelineFactory([passthrough], [None], output_file=new_file, data_field=data_field).filter_pipeline

    with ExitStack() as stack:
        hdf_files = [stack.enter_context(h5py.File(fd.file, 'r')) for fd in files]
        arrays = [hdf[data_field] for hdf in hdf_files]
        filters(arrays)

    fd = files[0]
    with h5py.File(fd.file, 'r') as fr, h5py.File(new_file, 'r+') as fw:
        fw[files[0].fs_field] = samp_res
        for k in set(fr.keys()) - {fd.fs_field, fd.data_field}:
            if hasattr(fr[k], 'shape') and not len(fr[k].shape):
                print('picked up key', k)
                fw[k] = fr[k][()]
    return new_file


def handoff_filter_hdf5_files(files, save_path, filters):
    """
    Filter several array files in hand-off mode (to match levels between recordings).

    Parameters
    ----------
    files: Sequence[FileData]
        The input HDF5 file data objects
    save_path: str
        Path to save output. Filtered files will have the same names as the input files.
    filters: FilterPipeline
        The FilterPipeline to apply. If None or empty, then return.

    """

    # first pass -- get total size and check for consistency
    _check_consistency(files)

    # TODO: Currently need to COPY the input data prior to filtering because HandOffIter output mode supports 'same'
    #  but does not yet support a matching set of array outputs.
    output_files = []
    for fd in files:
        save_file = os.path.join(save_path, os.path.split(fd.file)[1])
        with h5py.File(fd.file, 'r') as fr, h5py.File(save_file, 'w') as fw:
            # Copy basic fields for filtering
            fr.copy(fd.data_field, fw)
            fw[fd.fs_field] = fd.Fs
        output_files.append(save_file)

    if filters is not None and len(filters):
        filters = filters.make_pipeline(files[0].Fs, output_file='same').filter_pipeline

    with ExitStack() as stack:
        # Now opening HDF5 files in r+ mode to support write-back
        hdf_files = [stack.enter_context(h5py.File(file, 'r+')) for file in output_files]
        arrays = [hdf[fd.data_field] for fd, hdf in zip(files, hdf_files)]
        filters(arrays)

    for fd, outfile in zip(files, output_files):
        with h5py.File(fd.file, 'r') as fr, h5py.File(outfile, 'r+') as fw:
            for k in set(fr.keys()) - {fd.fs_field, fd.data_field}:
                if hasattr(fr[k], 'shape') and not len(fr[k].shape):
                    print('picked up key', k)
                    fw[k] = fr[k][()]


def batch_filter_hdf5_files(files, save_path, filters):

    for fd in files:
        save_file = os.path.join(save_path, os.path.split(fd.file)[1])
        print('input file:', fd.file, 'output file:', save_file)
        if filters is not None:
            fpipe = filters.make_pipeline(fd.Fs, output_file=save_file, data_field=fd.data_field).filter_pipeline
        else:
            fpipe = PipelineFactory([], None).filter_pipeline
        with h5py.File(fd.file, 'r') as fr:
            fpipe(fr[fd.data_field])
            # copy any other values from the last file?
            with h5py.File(save_file, 'a') as fw:
                # set down samp rate manually
                fw[fd.fs_field] = fd.Fs
                for k in set(fr.keys()) - {fd.fs_field, fd.data_field}:
                    if hasattr(fr[k], 'shape') and not len(fr[k].shape):
                        print('picked up key', k)
                        fw[k] = fr[k][()]
                    else:
                        fr.copy(k, fw)

        
class BatchFilesTool(PersistentWindow):
    file_dir = Directory
    _available_hdf5_files = List(Str)
    working_files = List([])
    join = Button('Join files')
    batch_proc = Button('Batch filter')
    downsample = Button('Downsample')
    batch_with_handoff = Bool(True)
    ds_rate = Int(10)
    filters = Instance(FilterPipeline, ())
    headstage_flavor = Enum(Mux7FileData, (Mux7FileData, OpenEphysFileData))

    file_size = Property(Str, depends_on='working_files, downsample')
    file_suffix = Str('')
    set_Fs = Property(Str, depends_on='working_files, downsample')

    @on_trait_change('working_files')
    def _foo(self):
        print(self.working_files)

    def _default_file_map(self):
        if not hasattr(self, '_file_map'):
            self._file_map = dict()
        klass = self.headstage_flavor
        # Rules:
        # 1) keep any existing file name to FileData map
        # 2) create new FileData map for novel keys
        # 3) remove unused/stale keys
        for f in self.working_files:
            if f in self._file_map:
                continue
            full_file = os.path.join(self.file_dir, f + '.h5')
            fd = klass(file=full_file)
            self._file_map[f] = fd
        for f in (set(self._file_map.keys()) - set(self.working_files)):
            self._file_map.pop(f)
        return self._file_map

    def _join_fired(self):
        if not len(self.working_files):
            return
        self._default_file_map()
        file_names = self.working_files
        file_objs = [self._file_map[f] for f in file_names]
        new_file_name = '-'.join(file_names)
        if self.file_suffix:
            new_file_name = new_file_name + '_' + self.file_suffix
        full_file = os.path.join(self.file_dir, new_file_name + '.h5')
        print('Concatenating {} to {}'.format([fd.file for fd in file_objs], full_file))
        concatenate_hdf5_files(file_objs, full_file, filters=self.filters)

    def _batch_proc_fired(self):
        new_dir = DirectoryDialog(
            new_directory=True,
            message='Choose a directory to save batch-processed files'
        )
        if new_dir.open() != OK:
            return
        if new_dir.path == self.file_dir:
            MessageDialog(message='Not letting you over-write the current directory!! Choose another').open()
            return
        file_names = self.working_files
        file_objs = [self._file_map[f] for f in file_names]
        print('Batch path:', new_dir.path)
        if self.batch_with_handoff:
            handoff_filter_hdf5_files(file_objs, new_dir.path, self.filters)
        else:
            batch_filter_hdf5_files(file_objs, new_dir.path, self.filters)
        self.filters.save_pipeline(os.path.join(new_dir.path, 'filters.txt'))

    def _downsample_fired(self):
        if not len(self.working_files):
            return
        klass = self.headstage_flavor
        self._default_file_map()
        for f in self.working_files:
            # use existing FileData if already mapped
            # (e.g. for sequential resampling)
            if f in self._file_map:
                fd = self._file_map[f]
            else:
                full_file = os.path.join(self.file_dir, f + '.h5')
                fd = klass(file=full_file)
            fd.ds_rate = self.ds_rate
            fd.file = fd.create_downsampled(where=self.file_dir)
            self._file_map[f] = fd

    @cached_property
    def _get_set_Fs(self):
        if not len(self.working_files):
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
            size = 2 ** (size - n * 10.0)
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
                        UItem('working_files',
                              editor=SetEditor(
                                  name='_available_hdf5_files',
                                  left_column_title='Available files',
                                  right_column_title='Files to process',
                                  ordered=True
                                  )
                            ),
                        UItem('filters', style='custom')
                    ),
                    HGroup(
                        VGroup(
                            HGroup(
                                UItem('batch_proc'),
                                UItem('join')
                            ),
                            HGroup(
                                UItem('batch_with_handoff'),
                                Label('Filter with level matching')
                            ),
                        ),
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
                    ),
                ),
                handler=FilesHandler,
                title='Batch processing tool',
                resizable=True
        )
        return v

