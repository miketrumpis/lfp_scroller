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

from traits.api import Range, Instance, Button, HasTraits, File, Float, Str
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Label

from ecogana.devices.electrode_pinouts import get_electrode_map


def h5mean(array, axis, block_size=20000):
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
        sl = ( range, slice(None) ) if axis else ( slice(None), range )
        mn[range] = array[sl].mean(axis)
        b += block_size
        if b >= mn_size:
            break
    return mn

            
class ReadCache(object):

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
        
    def updateHDF5Plot(self):
        if self.hdf5 is None:
            self.setData([])
            return
        
        #vb = self.getViewBox()
        vb = self.parentWidget()
        if vb is None:
            return  # no ViewBox yet
        
        # Determine what data range must be read from HDF5
        #xrange = vb.viewRange()[0]
        xrange = self.parentWidget().viewRange()[0]
        srange = map(lambda x: int(x / self._xscale), xrange)
        start = max(0,int(srange[0])-1)
        stop = min(self.hdf5.shape[-1], int(srange[1]+2))
        x_visible = np.arange(start, stop) * self._xscale
        
        # Decide by how much we should downsample 
        ds = int((stop-start) / self.limit) + 1

        if ds == 1:
            # Small enough to display with no intervention.
            visible = self.hdf5[self.index, start:stop]
            visible *= self._yscale
            visible += self.offset
            scale = 1
        else:
            # Here convert data into a down-sampled array suitable for visualizing.
            # Must do this piecewise to limit memory usage.        
            samples = 1 + ((stop-start) // ds)
            visible = np.zeros(samples*2, dtype=self.hdf5.dtype)
            x_samp = np.linspace(ds//2, len(x_visible)-ds//2, samples*2).astype('i')
            while x_samp[-1] >= len(x_visible):
                x_samp = x_samp[:-1]
            x_visible = np.take(x_visible, x_samp)
            sourcePtr = start
            targetPtr = 0
            
            # read data in chunks of ~1M samples
            chunkSize = (1000000//ds) * ds
            while sourcePtr < stop-1:
                sl = ( self.index,
                       slice(sourcePtr, min(stop,sourcePtr+chunkSize)) )
                chunk = self.hdf5[sl]
                sourcePtr += len(chunk)
                
                # reshape chunk to be integral multiple of ds
                chunk = chunk[:(len(chunk)//ds) * ds].reshape(len(chunk)//ds, ds)
                
                # compute max and min
                chunkMax = chunk.max(axis=1)
                chunkMin = chunk.min(axis=1)
                
                # interleave min and max into plot data to preserve envelope shape
                visible[targetPtr:targetPtr+chunk.shape[0]*2:2] = chunkMin
                visible[1+targetPtr:1+targetPtr+chunk.shape[0]*2:2] = chunkMax
                targetPtr += chunk.shape[0]*2
            
            visible = visible[:targetPtr]
            x_visible = x_visible[:targetPtr]
            visible *= self._yscale
            #visible += self.offset
            scale = ds * 0.5

        visible = detrend(visible, type='constant') + self.offset
        self.setData(x=x_visible, y=visible) # update the plot
        #self.setData(visible + self.index * offset)
        #self.setPos(start*self._xscale, 0) # shift to match starting index
        self.resetTransform()
        #self.scale(scale, 1)  # scale to match downsampling

#pg.mkQApp()

class FastScroller(object):

    def __init__(self, file_name, scale, spacing, chan_map,
                 load_channels='all', max_zoom=50000, units='mV'):
        

        #self.win = pg.GraphicsWindow()
        self.win = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(10,10,10))
        #sub_layout = layout.addLayout(row=0, col=0, rowspan=2, colspan=3)
        self.win.setCentralItem(layout)

        
        self.chan_map = chan_map

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
        self.p1 = layout.addPlot(colspan=2, row=0, col=0, rowspan=2)
        self.p1.enableAutoRange(False, False)
        self.p1.setLabel('left', 'Amplitude', units=units)
        self.p1.setLabel('bottom', 'Time', units='s')

        # The image panel
        self.p_img = layout.addViewBox(lockAspect=True, row=0, col=2)
        #self.p_img = layout.addPlot(col=4, rowspan=2, rowspan=2)
        self.img = pg.ImageItem(np.random.randn(*self.chan_map.geometry))
        self.p_img.addItem(self.img)
        self.p_img.autoRange()

        layout.nextRow()

        # The navigator plot
        self.p2 = layout.addPlot(row=2, col=0, colspan=3)
        self.p2.setLabel('left', 'Amplitude', units=units)
        self.p2.setLabel('bottom', 'Time', units='s')
        self.region = pg.LinearRegionItem() 
        self.region.setZValue(10)

        # Add the LinearRegionItem to the ViewBox,
        # but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        self.p2.addItem(self.region, ignoreBounds=True)

        self.p1.setAutoVisible(y=True)


        # Add traces to top plot
        f = h5py.File(file_name, 'r')
        nchan = f['data'].shape[0]
        del_t = f['Fs'].value ** -1.0
        array = ReadCache(f['data'])
        # array = ReadCache(f['data'][:, :100000])
        # scale = 1e3 / 20 # uV?

        self.p1.setXRange(0, 500*del_t)
        self.p1.vb.setLimits(maxXRange=max_zoom*del_t)

        if isinstance(load_channels, str) and load_channels.lower() == 'all':
            load_channels = xrange(nchan)

        self._curves = []
        for i in load_channels:
            curve = HDF5Plot()
            curve.setHDF5(array, i, spacing * i, scale=scale, dx=del_t)
            self.p1.addItem(curve)
            self._curves.append( curve )

        # Add mean trace to bottom plot
        mn = h5mean(f['data'], 0) * scale
        self.p2.plot(x=np.arange(len(mn))*del_t, y=mn)
        self.p2.setXRange(0, min(5e4, len(mn))*del_t)


        # Set bidirectional plot interaction
        self.region.sigRegionChanged.connect(self.update)
        self.p1.sigRangeChanged.connect(self.updateRegion)

        # Do navigation jumps (if shift key is down)
        #pg.SignalProxy(self.p2.scene().sigMouseClicked, slot=self.jump_nav)
        self.p2.scene().sigMouseClicked.connect(self.jump_nav)
        self.region.setRegion([0, 5000*del_t])

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
        
    def update(self):
        self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.p1.setXRange(minX, maxX, padding=0)    


    def updateRegion(self, window, viewRange):
        rgn = viewRange[0]
        self.region.setRegion(rgn)

def setup_qwidget_control(parent, editor, qwidget):
    #qwidget.setParent(parent)
    #parent.addWidget(qwidget)
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
            resizable=True,
            title='Quick Scanner'
        )
        return v

class VisLauncher(HasTraits):

    file = File
    b = Button
    scale = Float( 1e3 / 20 )
    offset = Float(0.5)
    chan_map = Str('ratv4_mux6')

    def _b_fired(self):
        if not os.path.exists(self.file):
            return
        chan_map, _ = get_electrode_map(self.chan_map)
        new_vis = FastScroller(self.file, self.scale, self.offset,
                               chan_map, load_channels=range(len(chan_map)))
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



