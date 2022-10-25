#!/usr/bin/env python
"""
Objects for loading/filtering HDF5 data streams and launching the
pyqtgraph / traitsui vis tool.
"""

import os
import numpy as np
import pandas as pd
import h5py
import pickle
from scipy.spatial.distance import pdist

from traits.api import Instance, Button, HasTraits, Float, on_trait_change, \
     Str, List, Enum, Int, Bool, cached_property, Range, Property, File
from traitsui.api import View, Group, VGroup, HGroup, UItem, Item, \
     Label, Handler, Tabbed, SetEditor
from pyface.api import MessageDialog

from ecogdata.devices.electrode_pinouts import get_electrode_map, electrode_maps, multi_arm_electrodes
from ecogdata.util import Bunch
from ecogdata.channel_map import ChannelMap
from ecogdata.expconfig import params

from ecoglib.signal_testing import interactive_mask

from .h5scroller import FastScroller
from .h5data import h5mean

from .data_files import Mux7FileData, OpenEphysFileData, SIOpenEphysFileData, BlackrockFileData, \
     FileData, BatchFilesTool, ActiveArrayFileData, RHDFileData, NWBFileData
from .filtering import FilterPipeline
from .new_scroller import VisWrapper
from .modules import ana_modules, default_modules


MEM_CAP = float(params.memory_limit)


class NoPickleError(Exception):
    pass


def find_pickled_map(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        if 'b_pickle' not in f.keys():
            raise NoPickleError
        arr = f['b_pickle'][0][:]
        if isinstance(arr, np.ndarray):
            b = pickle.loads(arr.tostring())
        else:
            b = pickle.loads(arr)
    chan_map = None
    for k in b.keys():
        if isinstance(b[k], ChannelMap):
            chan_map = b[k]
            break
    if chan_map is None:
        raise NoPickleError
    return chan_map


def map_from_table(csv_file):
    # fairly rigid
    table = pd.read_csv(csv_file, index_col=0).sort_values('channels')
    rows = table.rows
    cols = table.cols
    distances = pdist(np.c_[table.x_mm, table.y_mm])
    pitch = distances.min()
    m = ChannelMap.from_index(list(zip(rows, cols)), None, pitch=pitch)
    return m


def assign_file_from_headstage(headstage: str) -> FileData:
    headstage = headstage.lower()
    if headstage == 'mux3':
        fd = Mux7FileData(gain=10)
    elif headstage in ('mux5', 'mux6'):
        fd = Mux7FileData(gain=20)
    elif headstage == 'mux7':
        fd = Mux7FileData(gain=3)
    elif headstage == 'stim v4':
        fd = Mux7FileData(gain=4)
    elif headstage == 'si oephys':
        fd = SIOpenEphysFileData()
    elif headstage == 'intan (oephys)':
        fd = OpenEphysFileData()
    elif headstage == 'intan (rhd)':
        fd = RHDFileData()
    elif headstage == 'blackrock':
        fd = BlackrockFileData()
    elif headstage == 'active':
        fd = ActiveArrayFileData()
    elif headstage == 'nwb':
        fd = NWBFileData()
    else:
        fd = FileData()
    return fd


class HeadstageHandler(Handler):

    def object_headstage_changed(self, info):
        if not info.initialized and info.object.file_data:
            return
        hs = info.object.headstage.lower()
        info.object.file_data = assign_file_from_headstage(hs)
        # also update channel map for this case
        if hs in ('nwb', 'active'):
            info.object.chan_map = 'implicit'


_subset_chan_maps = ('psv_244_mux1', 'psv_244_mux3') + tuple(multi_arm_electrodes.keys())
# Short-cuts for two of the passive-244 permutations: all old/new headstages.
# For future additions -- format should be "map_name" + "/shortcutname" (to split on "/")
_subset_shortcuts = {'psv_244/newintan': ['intan64_new'] * 4,
                     'psv_244/oldintan': ['intan64'] * 4}
# append some non-standard channel maps for
# 1) active electrodes (these are constructed on the fly based on geometry)
# 2) unknown (can be built in the GUI)
# 3) a value that is set programmatically via the .set_chan_map attribute
available_chan_maps = sorted(electrode_maps.keys()) + \
                      sorted(multi_arm_electrodes.keys()) + \
                      list(_subset_shortcuts.keys()) + \
                      ['implicit', 'csv file', 'pickled', 'unknown', 'settable']


class VisLauncher(HasTraits):
    """
    Builds pyqtgraph/traitsui visualation using a timeseries
    from a FileData model that is filtered from a FilterPipeline.
    """
    
    file_data = Instance(FileData)
    filters = Instance(FilterPipeline, ())
    module_set = List(default_modules)
    all_modules = List(list(ana_modules.keys()))
    b = Button('Launch Visualization')
    offset = Enum(200, [0, 100, 200, 500, 1000, 2000, 5000])
    # max_window_width = Float(1200.0)
    _max_win = Property(Float, depends_on='file_data.file, file_data.data_field')
    _min_win = Int(30)
    max_window_width = Range(low='_min_win', high='_max_win', value=60)
    headstage = Enum('mux7',
                     ('mux3', 'mux5', 'mux6', 'mux7',
                      'stim v4', 'SI oephys', 'intan (oephys)', 'intan (rhd)',
                      'blackrock', 'active', 'NWB', 'unknown'))
    chan_map = Enum(available_chan_maps[0], available_chan_maps)
    loadable_map = File(filter=['*.csv'], exists=True)

    channel_data_order = Bool(False)
    n_chan = Int
    skip_chan = Str
    elec_geometry = Str
    chan_map_connectors = Str
    _is_subset_map = Property(fget=lambda self: self.chan_map in _subset_chan_maps)
    screen_channels = Bool(False)
    screen_start = Float(0)
    concat_tool_launch = Button('Launch Batch Tool')

    def __init__(self, **traits):
        super(VisLauncher, self).__init__(**traits)
        if self.headstage:
            self.file_data = assign_file_from_headstage(self.headstage)

    @cached_property
    def _get__max_win(self):
        if not self.file_data or not self.file_data.file or not self.file_data.data_field:
            return 100
        try:
            Fs = self.file_data.Fs
            array_size = self.file_data.array_size
            bytes_per_sec = Fs * array_size[0] * 8
            mx = int(MEM_CAP / bytes_per_sec)
            array_len = int(array_size[1] / Fs) + 1
            return min(array_len, mx)
        except (IOError, ValueError, TypeError, KeyError) as e:
            # these exceptions cover what may happen with a weird h5 file or incomplete setup
            return 100

    @on_trait_change('chan_map')
    def _set_default_connectors(self):
        # Only set a default if
        # 1) connections weren't already set (even for another map), and
        # 2) there are default connectors in the spec table
        if len(self.chan_map_connectors):
            return
        spec = multi_arm_electrodes.get(self.chan_map, None)
        if not spec:
            return
        self.chan_map_connectors = ', '.join(spec['default_daqs'])

    def _concat_tool_launch_fired(self):
        bft = BatchFilesTool()
        bft.edit_traits()
        # hold onto this reference or else window closes if idle?
        # self.__bft = bft
        
    def _get_screen(self, array, channels, chan_map, Fs):
        from ecogdata.expconfig import params
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
        screen_channels = [ channels[i] for i in range(len(mask)) if mask[i] ]
        screen_map = chan_map.subset(mask)
        return screen_channels, screen_map

    def launch(self):
        if not os.path.exists(self.file_data.file):
            return

        # Logic to normalize channel mapping
        # TODO: handle reference channels correctly
        if self.chan_map == 'unknown':
            try:
                nc = np.array( list(map(int, self.skip_chan.split(','))) )
            except:
                nc = []
            geo = list(map(int, self.elec_geometry.split(',')))
            n_sig_chan = self.n_chan - len(nc)
            chan_map = ChannelMap(np.arange(n_sig_chan), geo)
        elif isinstance(self.file_data, (NWBFileData, ActiveArrayFileData)):
            chan_map, nc = self.file_data.make_channel_map()
        elif self.chan_map in _subset_chan_maps:
            cnx = self.chan_map_connectors.split(',')
            cnx = [c.strip() for c in cnx]
            chan_map, nc, rf = get_electrode_map(self.chan_map, connectors=cnx)
            nc = list(set(nc).union(rf))
        elif self.chan_map in _subset_shortcuts:
            cnx = _subset_shortcuts[self.chan_map]
            map_name, shortcut = self.chan_map.split('/')
            chan_map, nc, rf = get_electrode_map(map_name, connectors=cnx)
            nc = list(set(nc).union(rf))
        elif self.chan_map == 'settable':
            chan_map = self.set_chan_map
            nc = []
        elif self.chan_map == 'pickled':
            try:
                chan_map = find_pickled_map(self.file_data.file)
                nc = []
            except NoPickleError:
                MessageDialog(message='No pickled ChannelMap').open()
                return
        elif self.chan_map == 'csv file':
            try:
                chan_map = map_from_table(self.loadable_map)
                nc = []
            except FileNotFoundError:
                MessageDialog(message='Select a CSV file in the browser').open()
                return
        else:
            chan_map, nc, rf = get_electrode_map(self.chan_map)
            nc = list(set(nc).union(rf))

        # # Check for transposed active data (coming from matlab)
        # if isinstance(self.file_data, ActiveArrayFileData) and self.file_data.is_transpose:
        #     self.file_data.create_transposed()
        #     print(self.file_data.file)
        x_scale = self.file_data.Fs ** -1
        num_vectors = len(chan_map) + len(nc)
        data_channels = [self.file_data.data_channels[i]
                         for i in range(num_vectors) if i not in nc]

        # permute channels to stack rows (unless data order is preferred)
        if not self.channel_data_order:
            chan_idx = list(zip(*chan_map.to_mat()))
            chan_order = chan_map.lookup(*list(zip(*sorted(chan_idx)[::-1])))
            data_channels = [data_channels[i] for i in chan_order]
            cls = type(chan_map)
            chan_map = cls([chan_map[i] for i in chan_order], chan_map.geometry, pitch=chan_map.pitch,
                           col_major=chan_map.col_major)

        filters = self.filters.make_pipeline(x_scale ** -1.0)
        array = self.file_data._compose_arrays(filters)
        if self.screen_channels:
            data_channels, chan_map = \
              self._get_screen(array, data_channels, chan_map, x_scale**-1.0)
            
        rowmask = self.file_data.data_channels
        nav = h5mean(array.file_array, 0, rowmask=rowmask)
        nav *= self.file_data.y_scale

        modules = [ana_modules[k] for k in self.module_set]
        new_vis = FastScroller(array, self.file_data.y_scale,
                               self.offset * 1e-6, chan_map, nav,
                               x_scale=x_scale,
                               load_channels=data_channels,
                               max_zoom=self.max_window_width)
        file_name = os.path.split(self.file_data.file)[1]
        file_name = os.path.splitext(file_name)[0]
        v_win = VisWrapper(new_vis, x_scale=x_scale, chan_map=chan_map,
                           y_spacing=self.offset, modules=modules, recording=file_name)
        view = v_win.default_traits_view()
        # TODO: it would be nice to be able to directly call launch() without first showing *this* object's panel
        view.kind = 'live'
        ui = v_win.edit_traits(view=view)
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
                    VGroup(Label('Channel map'), UItem('chan_map'),
                           UItem('loadable_map', enabled_when='chan_map=="csv file"')),
                    UItem('concat_tool_launch'),
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
                    VGroup(
                        Label('Connectors (comma-sep)'),
                        UItem('chan_map_connectors'),
                        visible_when='_is_subset_map'
                        )
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
                        Label('Offset per channel (in uV)'),
                        UItem('offset', )
                        ),
                    VGroup(
                        Label('Data order channels?'),
                        UItem('channel_data_order')
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
                HGroup(
                    UItem('b'),
                    Item('max_window_width', label='Max. window length')
                    )
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
