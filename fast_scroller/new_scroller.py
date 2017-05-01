# -*- coding: utf-8 -*-
"""
In this example we create a subclass of PlotCurveItem for displaying a very large 
data set from an HDF5 file that does not fit in memory. 

The basic approach is to override PlotCurveItem.viewRangeChanged such that it
reads only the portion of the HDF5 data that is necessary to display the visible
portion of the data. This is further downsampled to reduce the number of samples 
being displayed.

A more clever implementation of this class would employ some kind of caching 
to avoid re-reading the entire visible waveform at every update.
"""
from __future__ import division
import sys, os
import numpy as np
from scipy.signal import detrend
import h5py
import PySide
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph_extensions as pgx

from traits.api import Range, Instance, Button, HasTraits, File, Float, Str
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Label

from ecogana.devices.electrode_pinouts import get_electrode_map
from ecoglib.util import ChannelMap

def h5mean(array, axis, block_size=20000, start=0, stop=-1):
    """Compute mean of a 2D HDF5 array in blocks"""

    shape = array.shape
    if axis < 0:
        axis += len(shape)
    mn_size = shape[1 - axis]
    mn = np.empty(mn_size, 'd')
    b = 0
    while True:
        range = slice( b, min(mn_size, b + block_size) )
        # to compute block x, either slice like:
        # array[:, x].mean(0)
        # array[x, :].mean(1)
        if axis == 1:
            sl = ( range, slice(start, stop) )
        else:
            sl = ( slice(start, stop), range )
        mn[range] = array[sl].mean(axis)
        b += block_size
        if b >= mn_size:
            break
    return mn

            
class ReadCache(object):
    """
    Buffers row indexes from memmap or hdf5 file.

    For cases where array[0, n:m], array[1, n:m], array[2, n:m] are
    accessed sequentially, this object buffers the C x (m-n)
    submatrix before yielding individual rows.
    """
    
    def __init__(self, array):
        self._array = array
        self._current_slice = None
        self._current_seg = ()
        self.dtype = array.dtype
        self.shape = array.shape

    def __getitem__(self, sl):
        indx, range = sl
        if self._current_slice != range:
            all_sl = ( slice(None), range )
            self._current_seg = self._array[all_sl]
            self._current_slice = range
        # always return the full range after slicing with possibly
        # complex original range
        new_range = slice(None)
        new_sl = (indx, new_range)
        return self._current_seg[new_sl].copy()

class FilteredReadCache(ReadCache):
    """
    Apply row-by-row filters to a ReadCache
    """

    def __init__(self, array, filters):
        if not isinstance(filters, (tuple, list)):
            f = filters
            filters = [ f ] * len(array)
        self.filters = filters
        super(FilteredReadCache, self).__init__(array)

    def __getitem__(self, sl):
        idx = sl[0]
        x = super(FilteredReadCache, self).__getitem__(sl)
        return self.filters[idx]( x )

def make_filt(z):
    def _f(x):
        return x - z
    return _f
    
class DCOffsetReadCache(FilteredReadCache):
    """
    A filtered read cache with a simple offset subtraction.
    """

    def __init__(self, array, offsets):
        #filters = [lambda x: x - off for off in offsets]
        filters = [make_filt(off) for off in offsets]
        super(DCOffsetReadCache, self).__init__(array, filters)
        self.offsets = offsets

class HMSAxis(pg.AxisItem):

    def _to_hms(self, x):
        h = x // 3600
        m = (x - h*3600) // 60
        s = x - h * 3600 - m * 60
        return map(int, (h, m, s))
    
    def tickStrings(self, values, scale, spacing):
        strns = []
        if len(values)==0:
            return super(HMSAxis, self).tickStrings(values,scale,spacing)

        for x in values:
            strns.append('{0:02d}:{1:02d}:{2:02d}'.format(*self._to_hms(x)))
        return strns

        
class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.limit = 6000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        
    def setHDF5(self, data, index, offset, dx=1.0, scale=1):
        self.hdf5 = data
        self.index = index
        self._yscale = scale
        self._xscale = dx
        self.offset = offset
        self.updateHDF5Plot()
        
    def viewRangeChanged(self):
        self.updateHDF5Plot()

    def _view_range(self):
        vb = self.parentWidget()
        if vb is None:
            return  # no ViewBox yet
        # Determine what data range must be read from HDF5
        xrange = self.parentWidget().viewRange()[0]
        srange = map(lambda x: int(x / self._xscale), xrange)
        start = max(0,int(srange[0])-1)
        stop = min(self.hdf5.shape[-1], int(srange[1]+2))
        return start, stop
        
    def _set_data(self):
        r = self._view_range()
        if not r:
            return
        start, stop = r
        self.x_visible = np.arange(start, stop) * self._xscale
        self.y_visible = self.hdf5[self.index, start:stop]
        self.y_visible *= self._yscale
        
    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return
        
        self._set_data()
        r = self._view_range()
        if not r:
            return
        start, stop = r
        # Decide by how much we should downsample 
        ds = int((stop-start) / self.limit) + 1

        if ds == 1:
            # Small enough to display with no intervention.
            visible = self.y_visible
            x_visible = self.x_visible
            ## visible += self.offset
        else:
            # Here convert data into a down-sampled array suitable for visualizing.

            N = len(self.y_visible)
            Nsub = N // ds
            y = self.y_visible[:Nsub * ds].reshape(Nsub, ds)
            visible = np.empty(Nsub*2, dtype=self.hdf5.dtype)
            visible[0::2] = y.min(axis=1)
            visible[1::2] = y.max(axis=1)

            x_samp = np.linspace(ds//2, len(self.x_visible)-ds//2, Nsub*2).astype('i')
            if x_samp[-1] >= len(self.x_visible):
                x_samp[-1] = N-1
            x_visible = np.take(self.x_visible, x_samp)
             
        #visible = detrend(visible, type='constant') + self.offset
        visible = visible + self.offset
        self.setData(x=x_visible, y=visible) # update the plot
        #self.setPos(start*self._xscale, 0) # shift to match starting index
        self.resetTransform()

class FastScroller(object):

    def __init__(self, file_name, scale, spacing, chan_map,
                 load_channels='all', max_zoom=120.0, units='mV'):
        

        # Set up data access
        f = h5py.File(file_name, 'r')
        nchan = f['data'].shape[0]
        del_t = f['Fs'].value ** -1.0
        # array = ReadCache(f['data'][:, :100000])
        # scale = 1e3 / 20 # uV?
        mn_trace = h5mean(f['data'], 0) * scale
        mn_level = h5mean(f['data'], 1, start=int(3e4))
        if isinstance(load_channels, str) and load_channels.lower() == 'all':
            load_channels = xrange(nchan)
            self.chan_map = chan_map
        elif len(load_channels) <= len(chan_map):
            new_cm = ChannelMap( [chan_map[i] for i in load_channels],
                                  chan_map.geometry,
                                  col_major=chan_map.col_major )
            self.chan_map = new_cm
        else:
            raise ValueError('cannot map the listed channels')

        array = DCOffsetReadCache(f['data'], mn_level)
        self.array = array

        #self.win = pg.GraphicsWindow()
        self.win = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(10,10,10))
        #sub_layout = layout.addLayout(row=0, col=0, rowspan=2, colspan=3)
        self.win.setCentralItem(layout)
        layout.addLabel('Array heatmap')
        layout.addLabel('Zoomed plot')
        layout.addLabel('|')
        layout.nextRow()
        # win.setWindowTitle('pyqtgraph example: HDF5 big data')
        # set up a 3x3 grid:
        # __.__.__
        # |    .__|
        # |____.__|
        # |_______|

        # The focus plot
        #self.p1 = self.win.addPlot(row=0, col=0, rowspan=2, colspan=2)
        ## sub_layout.nextCol()
        ## sub_layout.nextCol()
        #layout.nextRow()
        sub_layout = layout.addLayout(colspan=1)
        self.p_img = sub_layout.addViewBox(lockAspect=True)

        ## self.img = pg.ImageItem(np.random.randn(*self.chan_map.geometry))
        self.img = pgx.ImageItem(np.random.randn(*self.chan_map.geometry).T)
        self.img.setLookupTable(pgx.get_colormap_lut(name='bipolar'))
        self.p_img.addItem(self.img)
        self.p_img.autoRange()
        self.img.setLevels( (-3, 3) )
        self.cb = pgx.ColorBarItem(
            image=self.img, label='Voltage ({0})'.format(units)
            )
        self.img.setLevels( (-10, 10) )
        sub_layout.addItem(self.cb)

        self.p1 = layout.addPlot(colspan=2, row=1, col=1)
        self.p1.enableAutoRange(False, False)
        self.p1.setLabel('left', 'Amplitude', units=units)
        self.p1.setLabel('bottom', 'Time', units='s')

        # The image panel
        layout.nextRow()

        # The navigator plot
        axis = HMSAxis(orientation='bottom')
        self.p2 = layout.addPlot(row=2, col=0, colspan=3,
                                 axisItems={'bottom':axis})
        self.p2.setLabel('left', 'Amplitude', units=units)
        self.p2.setLabel('bottom', 'Time')#, units='s')
        self.region = pg.LinearRegionItem() 
        self.region.setZValue(10)

        # Add the LinearRegionItem to the ViewBox,
        # but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        self.p2.addItem(self.region, ignoreBounds=True)

        self.p1.setAutoVisible(y=True)



        self.p1.setXRange(0, 500*del_t)
        self.p1.vb.setLimits(maxXRange=max_zoom)

        # curves will be itemized in order with the channel map
        self._curves = []
        for i in load_channels:
            curve = HDF5Plot()
            curve.setHDF5(array, i, spacing * i, scale=scale, dx=del_t)
            self.p1.addItem(curve)
            self._curves.append( curve )

        # Add mean trace to bottom plot
        self.p2.plot(x=np.arange(len(mn_trace))*del_t, y=mn_trace)
        self.p2.setXRange(0, min(5e4, len(mn_trace))*del_t)
        self.p2.setYRange(*np.percentile(mn_trace, [1, 99]))
        
        # Set bidirectional plot interaction
        self.region.sigRegionChanged.connect(self.update)
        self.p1.sigRangeChanged.connect(self.updateRegion)

        # Do navigation jumps (if shift key is down)
        #pg.SignalProxy(self.p2.scene().sigMouseClicked, slot=self.jump_nav)
        self.p2.scene().sigMouseClicked.connect(self.jump_nav)
        self.region.setRegion([0, 5000*del_t])

        # final adjustments to rows
        self.win.centralWidget.layout.setRowStretchFactor(0, 0.5)
        self.win.centralWidget.layout.setRowStretchFactor(1, 5)
        self.win.centralWidget.layout.setRowStretchFactor(2, 2.5)
        
    def jump_nav(self, evt):
        if QtGui.QApplication.keyboardModifiers() != QtCore.Qt.ShiftModifier:
            return
        pos = evt.scenePos()
        if not self.p2.sceneBoundingRect().contains(pos):
            return
        newX = self.p2.vb.mapSceneToView(pos).x()
        minX, maxX = self.region.getRegion()
        rng = 0.5 * (maxX - minX)
        self.region.setRegion( [ newX - rng, newX + rng ] )

    def set_mean_image(self):
        if not hasattr(self._curves[0], 'y_visible'):
            return
        image = np.empty( len(self.chan_map), 'd' )
        for i in xrange(len(self.chan_map)):
            image[i] = self._curves[i].y_visible.mean()
        self.img.setImage( self.chan_map.embed(image, fill=0).T,
                           autoLevels=False )
        
    def update(self):
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        # update zoom plot
        self.p1.setXRange(minX, maxX, padding=0)
        # update image with mean of the current view interval
        self.set_mean_image()
        

    def updateRegion(self, window, viewRange):
        rgn = viewRange[0]
        self.region.setRegion(rgn)

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    file = File
    b = Button
    r = Range(low=0, high=10, value=0)

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)

    def default_traits_view(self):
        v = View(
            VGroup(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow)),
                HGroup(
                    Item('b', label='Button'),
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

class VisLauncher(HasTraits):

    file = File
    b = Button
    scale = Float( 1e3 / 20 )
    offset = Float(0.5)
    max_window_width = Float(1200.0)
    chan_map = Str('ratv4_mux6')

    def _b_fired(self):
        if not os.path.exists(self.file):
            return
        chan_map, _ = get_electrode_map(self.chan_map)
        ii, jj = chan_map.to_mat()
        chan_order = np.argsort(ii)[::-1]
        new_vis = FastScroller(self.file, self.scale, self.offset,
                               chan_map, load_channels=chan_order,
                               max_zoom=self.max_window_width)
        v_win = VisWrapper(new_vis.win)
        view = v_win.default_traits_view()
        view.kind = 'livemodal'
        v_win.configure_traits(view=view)

    v = View(
        VGroup(
            Item('file', label='Visualize this file'),
            Label('Scales samples to voltage (default to mV for Mux7)'),
            UItem('scale'),
            Item('offset', label='Offset per channel (in mV)'),
            Item('chan_map', label='channel map name (use default)'),
            Item('b', label='Launch vis.')
            ),
        resizable=True,
        title='Launch Visualization'
    )
            
                     

#fileName = '/Users/mike/experiment_data/2017-04-25_hoffmann/test_002.h5'
#fileName = '/Users/mike/experiment_data/2013-11-01_Movshon_Lab/downsamp/m645r4#016_Fs3000.h5'
#fileName = '/Users/mike/experiment_data/2016-07-06_CSD/test_003.h5'

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    
    
    ## import sys
    ## if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    ##     QtGui.QApplication.instance().exec_()

    ## fileName = '/Users/mike/experiment_data/2017-04-28/test_007.h5'
    ## scale = 1e3 / 20
    ## chan_map, _ = get_electrode_map('ratv4_mux6')
    ## ## fs = FastScroller(fileName, scale, 0.5, chan_map,
    ## ##                   load_channels=range(60))

    ## #t = TestTraits(fs.win)
    ## t = TestTraits(None)
    ## #t.edit_traits()
    ## v = t.configure_traits()

    v = VisLauncher()
    v.configure_traits()



