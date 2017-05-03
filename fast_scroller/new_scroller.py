from __future__ import division
import sys
import numpy as np
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Instance, Button, HasTraits, \
     Enum, on_trait_change
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Group

from pyface.timer.api import Timer
import time

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 50, 100])
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)

    def __step_frame(self):
        n = self.__n
        x = self.__x
        y = self.__y
        if n >= self.__n_frames:
            self._atimer.Stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            self._atimer.Stop()
        PySide.QtCore.QCoreApplication.instance().processEvents()
        t_pause = scaled_dt - (time.time() - t0)
        if t_pause < 0:
            self.__f_skip += 1
            sys.stdout.flush()
        else:
            self._atimer.setInterval(t_pause * 1000.0)
        self.__n += self.__f_skip

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer') and self._atimer.IsRunning():
            self._atimer.Stop()
            return
        
        x, self.__y = self._qtwindow.current_data()
        self.__f_skip = 1
        self.__x = x[0]
        dt = self.__x[1] - self.__x[0]
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        print self.anim_time_scale * dt * 1000
        print self.__n_frames
        self._atimer = Timer( self.anim_time_scale * dt * 1000,
                              self.__step_frame )

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        v = View(
            VGroup(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow.win)),
                HGroup(
                    VGroup(
                        Item('anim_time_scale', label='Divide real time'),
                        Item('anim_frame'),
                        label='Animate Frames'
                        ),
                    VGroup(
                        UItem('y_spacing'),
                        label='Trace offset (mV)'
                        )
                    ),
                ),
            width=1200,
            height=500,
            resizable=True,
            title='Quick Scanner'
        )
        return v
    
## class FileHandler(Handler):

##     fields = List(Str)
##     def object_file_changed(self, info):
##         if not len(info.object.file):
##             self.fields = []
##             return
##         h5 = h5py.File(info.object.file, 'r')
##         self.fields = h5.keys()
##         h5.close()

## class FileData(HasTraits):
    
##     file = File
##     data_field = Str
##     fs_field = Str
##     y_scale = Float(1.0)
##     chan_map = Enum(electrode_maps.keys()[0], electrode_maps.keys())
##     zero_windows = Bool

##     def _compose_arrays(self, filter):
##         # default is just a wrapped array
##         h5 = h5py.File(self.file, 'r')
##         array = filter(h5[self.data_field])
##         mn_trace = h5mean(array, 0) * self.y_scale
##         if self.zero_windows:
##             array = FilteredReadCache(array, partial(detrend, type='constant'))
##         else:
##             array = ReadCache(array)
##         return array, mn_trace
    
##     def default_traits_view(self):
##         view = View(
##             VGroup(
##                 Item('file', label='Visualize this file'),
##                 HGroup(
##                     Item('data_field', label='Data field',
##                          editor=EnumEditor(name='handler.fields')),
##                     Item('fs_field', label='Fs field',
##                          editor=EnumEditor(name='handler.fields'))
##                     ),
##                 Label('Scales samples to voltage (default to mV for Mux7)'),
##                 UItem('y_scale'),
##                 Item('chan_map', label='channel map name'),
##                 Item('zero_windows', label='Remove local DC?')
##                 ),
##             handler=FileHandler,
##             resizable=True,
##             title='Launch Visualization'
##         )
##         return view
    
## class Mux7FileData(FileData):

##     file = File
##     y_scale = Float( 1e3 / 20 )
##     chan_map = Str('ratv4_mux6')
##     data_field = Property(fget=lambda self: 'data')
##     fs_field = Property(fget=lambda self: 'Fs')
    
##     def _compose_arrays(self, filter):
##         if self.zero_windows:
##             return super(Mux7FileData, self)._compose_arrays(filter)
##         f = h5py.File(self.file, 'r')
##         array = filter(f[self.data_field])
##         mn_level = h5mean(array, 1, start=int(3e4))
##         mn_trace = h5mean(array, 0) * self.y_scale
##         array = DCOffsetReadCache(array, mn_level)
##         return array, mn_trace

##     def default_traits_view(self):
##         view = View(
##             VGroup(
##                 Item('file', label='Visualize this file'),
##                 Label('Scales samples to voltage (default to mV for Mux7)'),
##                 UItem('y_scale'),
##                 Item('chan_map', label='channel map name (use default)'),
##                 Item('zero_windows', label='Remove local DC?')
##                 ),
##             ## resizable=True,
##             ## title='Launch Visualization'
##         )
##         return view

## class FilterMenu(HasTraits):
##     def default_traits_view(self):
##         t = self.traits()
##         #items = OrderedDict
##         items = list()
##         for k, v in t.items():
##             if k in ('trait_added', 'trait_modified'):
##                 continue
##             items.append( Item(k, label=v.info_text, style='simple') )
##         return View(*items, resizable=True)
    
## class BandpassMenu(FilterMenu):

##     lo = Float(-1, info_text='low freq cutoff (-1 for none)')
##     hi = Float(-1, info_text='high freq cutoff (-1 for none)')
##     ord = Int(5, info_text='filter order')

## class ButterMenu(BandpassMenu):
##     def make_filter(self, Fs):
##         return butter_bp(lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord)

## class Cheby1Menu(BandpassMenu):
##     ripple = Float(0.1, info_text='passband ripple (dB)')
##     def make_filter(self, Fs):
##         return cheby1_bp(
##             self.ripple, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
##             )

## class Cheby2Menu(BandpassMenu):
##     stop_atten = Float(40, info_text='stopband attenuation (dB)')
##     def make_filter(self, Fs):
##         return cheby2_bp(
##             self.ripple, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
##             )
    
## class NotchMenu(FilterMenu):
##     notch = Float(60, info_text='notch frequency')
##     nwid = Float(3.0, info_text='notch width (Hz)')
##     nzo = Int(3, info_text='number of zeros at notch')
##     def make_filter(self, Fs):
##         return notch(self.notch, Fs, self.nwid, nzo=self.nzo)
    
## class FilterHandler(Handler):

##     def object_filt_type_changed(self, info):
##         if info.object.filt_type == 'butterworth':
##             info.object.filt_menu = ButterMenu()
##         elif info.object.filt_type == 'cheby 1':
##             info.object.filt_menu = Cheby1Menu()
##         elif info.object.filt_type == 'cheby 2':
##             info.object.filt_menu = Cheby2Menu()
##         elif info.object.filt_type == 'notch':
##             info.object.filt_menu = NotchMenu()


## class Filter(HasTraits):
##     filt_type = Enum( 'butterworth', ['butterworth', 'cheby 1',
##                                       'cheby 2', 'notch'] )

##     filt_menu = Instance(HasTraits)

##     view = View(
##         VGroup(
##             Item('filt_type', label='Filter type'),
##             Group(
##                 UItem('filt_menu', style='custom'),
##                 label='Filter menu'
##                 )
##             ),
##         handler=FilterHandler
##         )
    
## filter_table = TableEditor(
##     columns = [ ObjectColumn( name='filt_type' ) ],
##     deletable=True,
##     auto_size=True,
##     show_toolbar=True,
##     reorderable=True,
##     edit_view=None,
##     row_factory=Filter
##     )

## from tempfile import NamedTemporaryFile
## def pipeline_factory(f_list):
##     if not len(f_list):
##         return lambda x: x
##     def copy_x(x):
##         with NamedTemporaryFile(mode='ab', dir='.') as f:
##             print f.name
##             fw = h5py.File(f.name, 'w')
##             y = fw.create_dataset('data', shape=x.shape,
##                                   dtype=x.dtype, chunks=x.chunks)
##         return y
        
##     def run_list(x, axis=1):
##         y = copy_x(x)
##         b, a = f_list[0]
##         bfilter(b, a, x, axis=axis, out=y)
##         for (b, a) in f_list[1:]:
##             bfilter(b, a, y, axis=axis)
##         return y

##     return run_list
        

## class FilterPipeline(HasTraits):
##     filters = List(Filter)

##     def make_pipeline(self, Fs):
##         filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
##         return pipeline_factory(filters)
    
##     view = View(
##         Group(
##             UItem('filters', editor=filter_table),
##             show_border=True
##             ),
##         title='Filters',
##         )

## class VisLauncher(HasTraits):
##     file_data = Instance(FileData)
##     filters = Instance(FilterPipeline)
##     b = Button
##     offset = Float(0.5)
##     max_window_width = Float(1200.0)

##     def __init__(self, **traits):
##         super(VisLauncher, self).__init__(**traits)
##         self.add_trait('filters', FilterPipeline())
##         self.add_trait('tabs', List( [self.file_data, self.filters] ))

##     def _b_fired(self):
##         if not os.path.exists(self.file_data.file):
##             return
        
##         chan_map, _ = get_electrode_map(self.file_data.chan_map)
##         ii, jj = chan_map.to_mat()
##         chan_order = np.argsort(ii)[::-1]
##         h5 = h5py.File(self.file_data.file, 'r')
##         x_scale = h5[self.file_data.fs_field].value ** -1.0
##         h5.close()
##         filters = self.filters.make_pipeline(x_scale ** -1.0)
##         array, nav = self.file_data._compose_arrays(filters)
##         new_vis = FastScroller(array, self.file_data.y_scale,
##                                self.offset, chan_map, nav,
##                                x_scale=x_scale,
##                                load_channels=chan_order,
##                                max_zoom=self.max_window_width)
##         v_win = VisWrapper(new_vis)
##         view = v_win.default_traits_view()
##         view.kind = 'livemodal'
##         v_win.configure_traits(view=view)

        
##     def default_traits_view(self):
##         v = View(
##             VGroup(
##                 UItem(
##                     'tabs', style='custom',
##                     editor=ListEditor(
##                         use_notebook=True, deletable=False,
##                         dock_style='tab'
##                         )
##                     ),
##                 Item('offset', label='Offset per channel (in mV)'),
##                 Item('b', label='Launch vis.'),
##             ),
##             resizable=True,
##             title='Launch Visualization'
##         )
##         return v

    
## if __name__ == '__main__':
    
##     ## fd = Mux7FileData()
##     fd = FileData()
##     v = VisLauncher(file_data=fd)
##     ## v = VisLauncher()
##     ## v.configure_traits()


##     ## v = VisLauncher(file='/Users/mike/experiment_data/Viventi 2017-04-10 P4 Rat 17009/2017-04-10_16-21-18_010_Fs1000.h5',
##     ##                 data_field='chdata',
##     ##                 fs_field='Fs',
##     ##                 chan_map='psv_61_intan2',
##     ##                 y_scale=0.000198)

##     v.configure_traits()
                    
