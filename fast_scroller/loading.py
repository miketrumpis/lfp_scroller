#!/usr/bin/env python
"""
Objects for loading/filtering HDF5 data streams and launching the
pyqtgraph / traitsui vis tool.
"""

from __future__ import division
import os
import numpy as np
import h5py

from traits.api import Instance, Button, HasTraits, Float, \
     Str, List, Enum, Int, Bool, cached_property, Range, Property
from traitsui.api import View, Group, VGroup, HGroup, UItem, Item, \
     Label, Handler, Tabbed, SetEditor

from ecogana.devices.electrode_pinouts import get_electrode_map, electrode_maps
from ecogana.devices.channel_picker import interactive_mask
from ecoglib.util import ChannelMap, Bunch
from ecogana.expconfig import params

from .h5scroller import FastScroller
from .h5data import h5mean

from .data_files import Mux7FileData, OpenEphysFileData, BlackrockFileData, \
     FileData, BatchFilesTool, ActiveArrayFileData
from .filtering import FilterPipeline
from .new_scroller import VisWrapper
from .modules import ana_modules, default_modules

MEM_CAP = float(params.memory_limit)

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
            fd = Mux7FileData(gain=3)
        elif hs.lower() == 'stim v4':
            fd = Mux7FileData(gain=4)
        elif hs.lower() == 'intan':
            fd = OpenEphysFileData()
        elif hs.lower() == 'blackrock':
            fd = BlackrockFileData()
        elif hs.lower() == 'active':
            fd = ActiveArrayFileData()
            info.object.chan_map = 'active'
        else:
            fd = FileData()
        info.object.file_data = fd

_subset_chan_maps = ('psv_244_mux1', 'psv_244_mux3')
# append some non-standard channel maps for
# 1) active electrodes (these are constructed on the fly based on geometry)
# 2) unknown (can be built in the GUI)
# 3) a value that is set programmatically via the .set_chan_map attribute
available_chan_maps = sorted(electrode_maps.keys()) + \
  ['active', 'unknown', 'settable']
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
    #max_window_width = Float(1200.0)
    _max_win = Property(Float, depends_on='file_data.file')
    _min_win = Int(30)
    max_window_width = Range(low='_min_win', high='_max_win', value=60)
    headstage = Enum('mux7',
                     ('mux3', 'mux5', 'mux6', 'mux7',
                      'stim v4', 'intan', 'blackrock', 'active', 'unknown'))
    chan_map = Enum(available_chan_maps[0], available_chan_maps)

    n_chan = Int
    skip_chan = Str
    elec_geometry = Str
    chan_map_connectors = Str
    _is_subset_map = Property(
        fget=lambda self: self.chan_map in _subset_chan_maps
        )
    screen_channels = Bool(False)
    screen_start = Float(0)
    concat_tool_launch = Button('Launch Batch Tool')

    def __init__(self, **traits):
        super(VisLauncher, self).__init__(**traits)
        self.add_trait('filters', FilterPipeline())

    @cached_property
    def _get__max_win(self):
        if not self.file_data or not self.file_data.file:
            return 100
        try:
            with h5py.File(self.file_data.file, 'r') as h5:
                #Fs = h5[self.file_data.fs_field].value
                Fs = self.file_data.Fs
                array_size = h5[self.file_data.data_field].shape
            bytes_per_sec = Fs * array_size[0] * 8
            mx = int(MEM_CAP / bytes_per_sec)
            array_len = int(array_size[1] / Fs) + 1
            return min(array_len, mx)
        except (IOError, ValueError) as e:
            return 100
    
    def _concat_tool_launch_fired(self):
        bft = BatchFilesTool()
        bft.edit_traits()
        # hold onto this reference or else window closes if idle?
        self.__bft = bft
        
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

        # Logic to normalize channel mapping
        if self.chan_map == 'unknown':
            try:
                nc = np.array( map(int, self.skip_chan.split(',')) )
            except:
                nc = []
            geo = map(int, self.elec_geometry.split(','))
            n_sig_chan = self.n_chan - len(nc)
            chan_map = ChannelMap(np.arange(n_sig_chan), geo)
        elif self.chan_map == 'active':
            chan_map, nc = self.file_data.make_channel_map()
        elif self.chan_map in _subset_chan_maps:
            cnx = self.chan_map_connectors.split(',')
            cnx = [c.strip() for c in cnx]
            chan_map, nc = get_electrode_map(self.chan_map,
                                             connectors=cnx)
        elif self.chan_map == 'settable':
            chan_map = self.set_chan_map
            nc = []
        else:
            chan_map, nc = get_electrode_map(self.chan_map)

        # Check for transposed active data (coming from matlab)
        if isinstance(self.file_data, ActiveArrayFileData) and self.file_data.is_transpose:
            self.file_data.create_transposed()
            print self.file_data.file

        with h5py.File(self.file_data.file, 'r') as h5:
            #x_scale = h5[self.file_data.fs_field].value ** -1.0
            x_scale = self.file_data.Fs ** -1.0
            array_size = h5[self.file_data.data_field].shape[0]
        num_vectors = len(chan_map) + len(nc)

        data_channels = [self.file_data.data_channels[i]
                         for i in xrange(num_vectors) if i not in nc]

        # permute  channels to stack rows
        chan_idx = zip(*chan_map.to_mat())
        chan_order = chan_map.lookup(*zip( *sorted(chan_idx)[::-1] ))
        data_channels = [data_channels[i] for i in chan_order]
        chan_map = ChannelMap( [chan_map[i] for i in chan_order],
                               chan_map.geometry, pitch=chan_map.pitch,
                               col_major=chan_map.col_major )

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
                               self.offset * 1e-3, chan_map, nav,
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
                    Group(
                        Item('chan_map_connectors',
                             label='Connectors (comma-sep)'),
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
