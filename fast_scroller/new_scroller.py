from __future__ import division
import sys, os
import numpy as np
import h5py
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Range, Instance, Button, HasTraits, File, Float, \
     Str, Property, List, Enum
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Label, Handler, EnumEditor

from ecogana.devices.electrode_pinouts import get_electrode_map, electrode_maps

from h5scroller import h5mean, ReadCache, FastScroller, DCOffsetReadCache

from pyface.timer.api import Timer
import time

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    file = File
    anim_frame = Button('Animate')
    stop_anim = Button('Stop')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 50, 100])
    r = Range(low=0, high=10, value=0)

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)

    def _anim_frame_fired(self):
        x, y = self._qtwindow.current_data()
        f_skip = 1
        x = x[0]
        dt = x[1] - x[0]
    
        scaled_dt = self.anim_time_scale * dt
        n_frames = y.shape[1]
        print 'got', n_frames, 'frames'
        print 'pause', scaled_dt, 'sec between frames'
        n = 0
        # Step through frames, pausing enough time to achieve
        # correct frame rate. If going too slow, start skipping frames.
        app = PySide.QtCore.QCoreApplication.instance()
        while n <= n_frames:
            t0 = time.time()
            self._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
            app.processEvents()
            t_pause = scaled_dt - (time.time() - t0)
            print t_pause
            if t_pause < 0:
                f_skip += 1
            else:
                time.sleep(t_pause)
            n += f_skip

    def default_traits_view(self):
        v = View(
            VGroup(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow.win)),
                HGroup(
                    VGroup(
                        Item('anim_time_scale', label='Divide real time'),
                        HGroup(Item('anim_frame'), Item('stop_anim')),
                        label='Animate Frames'
                        ),
                    Item('r', label='Range'),
                    label='Toolbar'
                    ),
                ),
            width=1200,
            height=500,
            resizable=True,
            title='Quick Scanner'
        )
        return v

class FileHandler(Handler):

    fields = List(Str)
    def object_file_changed(self, info):
        if not len(info.object.file):
            self.fields = []
            return
        h5 = h5py.File(info.object.file, 'r')
        self.fields = h5.keys()
        h5.close()
    
class VisLauncher(HasTraits):
    file = File
    data_field = Str
    fs_field = Str
    
    b = Button
    y_scale = Float(1.0)
    offset = Float(0.5)
    max_window_width = Float(1200.0)
    chan_map = Enum(electrode_maps.keys()[0], electrode_maps.keys())

    def _compose_arrays(self):
        # default is just a wrapped array
        h5 = h5py.File(self.file, 'r')
        mn_trace = h5mean(h5[self.data_field], 0) * self.y_scale
        return ReadCache(h5[self.data_field]), mn_trace
    
    def _b_fired(self):
        if not os.path.exists(self.file):
            return
        
        chan_map, _ = get_electrode_map(self.chan_map)
        ii, jj = chan_map.to_mat()
        chan_order = np.argsort(ii)[::-1]
        h5 = h5py.File(self.file, 'r')
        x_scale = h5[self.fs_field].value ** -1.0
        h5.close()
        array, nav = self._compose_arrays()
        new_vis = FastScroller(array, self.y_scale, self.offset,
                               chan_map, nav, x_scale=x_scale,
                               load_channels=chan_order,
                               max_zoom=self.max_window_width)
        v_win = VisWrapper(new_vis)
        view = v_win.default_traits_view()
        view.kind = 'livemodal'
        v_win.configure_traits(view=view)

    view = View(
        VGroup(
            Item('file', label='Visualize this file'),
            HGroup(
                Item('data_field', label='Data field',
                     editor=EnumEditor(name='handler.fields')),
                Item('fs_field', label='Fs field',
                     editor=EnumEditor(name='handler.fields'))
                ),
            Label('Scales samples to voltage (default to mV for Mux7)'),
            UItem('y_scale'),
            Item('offset', label='Offset per channel (in mV)'),
            Item('chan_map', label='channel map name'),
            Item('b', label='Launch vis.')
            ),
        handler=FileHandler,
        resizable=True,
        title='Launch Visualization'
    )
    
class Mux7VisLauncher(VisLauncher):

    file = File
    b = Button
    y_scale = Float( 1e3 / 20 )
    offset = Float(0.5)
    max_window_width = Float(1200.0)
    chan_map = Str('ratv4_mux6')
    data_field = Property(fget=lambda self: 'data')
    fs_field = Property(fget=lambda self: 'Fs')
    
    def _compose_arrays(self):
        f = h5py.File(self.file, 'r')
        mn_level = h5mean(f[self.data_field], 1, start=int(3e4))
        mn_trace = h5mean(f[self.data_field], 0) * self.y_scale
        array = DCOffsetReadCache(f[self.data_field], mn_level)
        return array, mn_trace
    
    view = View(
        VGroup(
            Item('file', label='Visualize this file'),
            Label('Scales samples to voltage (default to mV for Mux7)'),
            UItem('y_scale'),
            Item('offset', label='Offset per channel (in mV)'),
            Item('chan_map', label='channel map name (use default)'),
            Item('b', label='Launch vis.')
            ),
        resizable=True,
        title='Launch Visualization'
    )

if __name__ == '__main__':
    
    #v = Mux7VisLauncher()
    v = VisLauncher()
    v.configure_traits()



