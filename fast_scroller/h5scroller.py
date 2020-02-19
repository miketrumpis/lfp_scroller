from contextlib import contextmanager
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
from .pyqtgraph_extensions import ImageItem, ColorBarItem, get_colormap_lut
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecogdata.parallel.array_split import timestamp


def embed_frame(channel_map, vector):
    if isinstance(channel_map, CoordinateChannelMap):
        frame = channel_map.embed(vector, interpolate='cubic')
        frame = frame.filled(0)[::-1].copy()
    else:
        frame = channel_map.embed(vector, fill=0)[::-1].copy()
    return frame


@contextmanager
def block_signals(qtobj, forced_state=None):
    """
    Context that blocks/unblocks signal emission. By default the signal emission state is toggled, but it can be
    forced to a state using "forced_state". Emission status is always returned to the original status, regardless of
    forcing.

    Parameters
    ----------
    qtobj: QtCore.QObject
    forced_state: bool or None

    """
    if forced_state is not None:
        state = not forced_state
    else:
        state = qtobj.signalsBlocked()
    try:
        orig_state = qtobj.blockSignals(not state)
        yield
    finally:
        qtobj.blockSignals(orig_state)


class HMSAxis(pg.AxisItem):

    def _to_hms(self, x):
        h = x // 3600
        m = (x - h*3600) // 60
        s = x - h * 3600 - m * 60
        return list(map(int, (h, m, s)))
    
    def tickStrings(self, values, scale, spacing):
        strns = []
        if len(values)==0:
            return super(HMSAxis, self).tickStrings(values,scale,spacing)

        for x in values:
            strns.append('{0:02d}:{1:02d}:{2:02d}'.format(*self._to_hms(x)))
        return strns


class PlainSecAxis(pg.AxisItem):

    def tickStrings(self, values, scale, spacing):
        if len(values)==0:
            return super(PlainSecAxis, self).tickStrings(values,scale,spacing)
        strns = list(map(lambda x: str(x), values))
        return strns


class CurveCollection(QtCore.QObject):
    data_changed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, curves, *args, **kwargs):
        super(CurveCollection, self).__init__(*args, **kwargs)
        self.curves = curves
        self._current_set = set()
        self._connections = list()
        for c in curves:
            self._connections.append(c.sigPlotChanged.connect(self.count_curves))

    def count_curves(self, curve):
        self._current_set.add(curve)
        if len(self._current_set) == len(self.curves):
            self.data_changed.emit(self)
            print('reached count so emitting and resetting current set')
            self._current_set = set()

    def current_data(self):
        x_vis = self.curves[0].x_visible
        y_vis = self.curves[0].y_visible
        x = np.empty((len(self.curves), len(x_vis)), x_vis.dtype)
        y = np.empty((len(self.curves), len(y_vis)), y_vis.dtype)
        x[0] = x_vis
        y[0] = y_vis
        for i in range(1, len(self.curves)):
            x[i] = self.curves[i].x_visible
            y[i] = self.curves[i].y_visible
        return x, y

    @contextmanager
    def disable_listener(self, blocking=True):
        if blocking:
            for cnx, curve in zip(self._connections, self.curves):
                curve.sigPlotChanged.disconnect(cnx)
            self._connections = list()
        try:
            yield
        finally:
            if blocking:
                for c in self.curves:
                    self._connections.append(c.sigPlotChanged.connect(self.count_curves))

    def set_data_on_collection(self, array, ignore_signals=True):
        for data, line in zip(array, self.curves):
            with self.disable_listener(blocking=ignore_signals):
                line.set_external_data(data)

    def redraw_data(self, ignore_signals=False):
        # this would be used e.g. when the y-offsets have changed but the view limits have not
        for line in self.curves:
            with self.disable_listener(blocking=ignore_signals):
                line.updateHDF5Plot()


class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        self.limit = 6000 # maximum number of samples to be plotted
        pg.PlotCurveItem.__init__(self, *args, **kwds)
        # This item also has a curve point and (in)visible text
        #self.curve_point = pg.CurvePoint(self)
        self.text = pg.TextItem('empty', anchor=(0.5, 1.0), color=(255, 98, 65), fill=(65, 222, 255, 180))
        #self.text.setParent(self.curve_point)
        self.text.setVisible(False)
        self.series_filter = None
        # track current values of (x1, x2) to protect against reloads triggered by spurious view changes
        self._cx1 = 0
        self._cx2 = 0
        
    def setHDF5(self, data, index, offset, dx=1.0, scale=1):
        self.hdf5 = data
        self.index = index
        self._yscale = scale
        self._xscale = dx
        self.offset = offset
        self.updateHDF5Plot()

    def _view_really_changed(self):
        r = self._view_range()
        if r is None:
            return False
        return (r[0] != self._cx1) or (r[1] != self._cx2)

    def viewRangeChanged(self):
        if self.offset == 0:
            print('range-change-triggered', timestamp())
        # only look for x-range changes that actually require loading different data
        if self._view_really_changed():
            if self.offset == 0:
                print('ranged actually changed:', self._view_range(), timestamp())
            self.updateHDF5Plot()

    def _view_range(self):
        vb = self.parentWidget()
        if vb is None:
            return  # no ViewBox yet
        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        srange = list(map(lambda x: int(x / self._xscale), xrange))
        start = max(0,int(srange[0])-1)
        stop = min(self.hdf5.shape[-1], int(srange[1]+2))
        return start, stop
        
    def _set_data(self):
        r = self._view_range()
        if not r:
            return
        start, stop = r
        self.x_visible = np.arange(start, stop) * self._xscale
        self.y_visible = self.hdf5[self.index, start:stop] * self._yscale

    def set_external_data(self, y):
        if len(y) != len(self.x_visible):
            raise ValueError('Shape of external series does not match the current window.')
        self.y_visible = y
        self.updateHDF5Plot(data_ready=True)
        
    def updateHDF5Plot(self, data_ready=False):
        if self.offset == 0:
            print('update hdf5 lines called: DR', data_ready, timestamp())
        if self.hdf5 is None:
            self.setData([])
            return

        if not data_ready:
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
        else:
            # Here convert data into a down-sampled
            # array suitable for visualizing.

            N = len(self.y_visible)
            Nsub = max(5, N // ds)
            y = self.y_visible[:Nsub * ds].reshape(Nsub, ds)
            visible = np.empty(Nsub*2, dtype='d') #dtype=self.hdf5.dtype)
            visible[0::2] = y.min(axis=1)
            visible[1::2] = y.max(axis=1)

            x_samp = np.linspace(ds//2, len(self.x_visible)-ds//2, Nsub*2).astype('i')
            if x_samp[-1] >= len(self.x_visible):
                x_samp[-1] = N-1
            x_visible = np.take(self.x_visible, x_samp)

        visible = visible + self.offset
        self.setData(x=x_visible, y=visible) # update_zoom_callback the plot
        #self.setPos(start*self._xscale, 0) # shift to match starting index
        self.resetTransform()
        self.set_text_position()
        # mark these start, stop positions to match against future view-change signals
        self._cx1 = start
        self._cx2 = stop

    def set_text_position(self):
        if self.text.isVisible():
            x_visible = self.x_visible
            y_visible = self.y_visible + self.offset
            # put text 20% from the left
            x_pos = 0.8 * x_visible[0] + 0.2 * x_visible[-1]
            # set height to the max in a 10% window around x_pos
            wid = max(1, int(0.05 * len(y_visible)))
            n_pos = int(0.2 * len(y_visible))
            y_pos = y_visible[n_pos - wid:n_pos + wid].max()
            self.text.setPos(x_pos, y_pos)

    def toggle_text(self, i, j):
        self.text.setText('({0}, {1})'.format(i, j))
        vis = self.text.isVisible()
        self.text.setVisible(not vis)
        self.set_text_position()
        if not vis:
            pen = pg.mkPen(color='c', width=4)
        else:
            pen = pg.mkPen(width=0)
        self.setShadowPen(pen)


class FastScroller(object):

    # noinspection PyStatementEffect
    def __init__(
            self, array, y_scale, y_spacing, chan_map,
            nav_trace, x_scale=1, load_channels='all',
            max_zoom=120.0, units='V'
            ):
        """
        Scroller GUI loading a minimal amount of timeseries to display.

        X-units are currently assumed to be seconds.

        Parameters
        ----------
        array : array-like
            Array-like access to a mapped array (i.e. can index with
            array[i, range] slices).
        y_scale : float
            Scale the raw samples to units.
        y_spacing : float
            Spacing for the channel traces (in units)
        chan_map : ChannelMap
            Matrix map of the channels.
        nav_trace : array-like (1D)
            Navigation trace that is the same length as the channel
            timeseries. If not given, then mean of channels is used.
            **THIS MAY BE A COSTLY COMPUTATION**
        x_scale : float (optional)
            X-axis scale (e.g. sampling interval)
        load_channels : sequence | 'all'
            Order of (subset of) channels to load. Otherwise load all in array.
        units : str
            Y-axis units

        """

        nchan = array.shape[0]
        if isinstance(load_channels, str) and load_channels.lower() == 'all':
            load_channels = range(nchan)
            self.chan_map = chan_map
        elif len(load_channels) == len(chan_map):
            self.chan_map = chan_map
        elif len(load_channels) < len(chan_map):
            # "load_channels" indexes an array of recording channels
            # that may include grounded channels or other junk.
            raise NotImplementedError('cannot yet subset data channels')
            # new_cm = ChannelMap( [chan_map[i] for i in load_channels],
            #                       chan_map.geometry,
            #                       col_major=chan_map.col_major )
            # self.chan_map = new_cm
        else:
            raise ValueError('cannot map the listed channels')

        self.array = array
        self.y_scale = y_scale
        self.load_channels = load_channels

        # The main window + layout
        self.win = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(10,10,10))
        self.win.setCentralItem(layout)
        # Adding columns to layout: just titles on the top row
        layout.addLabel('Array heatmap')
        layout.addLabel('Zoomed plot')
        layout.addLabel('|')
        # Next row has 1) the heatmap image with the colorbar widget
        layout.nextRow()
        sub_layout = layout.addLayout(colspan=1)
        self.p_img = sub_layout.addViewBox(lockAspect=True)

        self.img = ImageItem(np.random.randn(*self.chan_map.geometry))
        self.img.setLookupTable(get_colormap_lut(name='bipolar'))
        self.p_img.addItem(self.img)
        self.p_img.autoRange()
        self.cb = ColorBarItem(image=self.img, label='Voltage ({0})'.format(units))
        self.img.setLevels( (-y_spacing, y_spacing) )
        sub_layout.addItem(self.cb)
        mid_x, top_y = self.chan_map.geometry[1] / 2.0, self.chan_map.geometry[0] + 2.0
        # add a text label on top of the box ("anchor" has text box is centered on its x, y position)
        self.frame_text = pg.TextItem('empty', anchor=(0.5, 0.5), color=(255, 255, 255))
        self.frame_text.setPos(mid_x, top_y)
        self.p_img.addItem(self.frame_text)

        # 2) the stacked traces plot (colspan 2)
        axis = PlainSecAxis(orientation='bottom')
        self.p1 = layout.addPlot(colspan=2, row=1, col=1,
                                 axisItems={'bottom':axis})
        self.p1.enableAutoRange(False, False)
        self.p1.setLabel('left', 'Amplitude', units=units)
        self.p1.setLabel('bottom', 'Time')

        # Next layout row has the navigator plot (colspan 3)
        layout.nextRow()

        # The navigator plot
        axis = HMSAxis(orientation='bottom')
        self.p2 = layout.addPlot(row=2, col=0, colspan=3,
                                 axisItems={'bottom':axis})
        self.p2.setLabel('left', 'Amplitude', units=units)
        self.p2.setLabel('bottom', 'Time')
        self.region = pg.LinearRegionItem() 
        self.region.setZValue(10)

        # Add the LinearRegionItem to the ViewBox,
        # but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        self.p2.addItem(self.region, ignoreBounds=True)

        self.p1.setAutoVisible(y=True)
        self.p1.setXRange(0, 500*x_scale)
        self.p1.vb.setLimits(maxXRange=max_zoom)

        # curves will be itemized in order with the channel map
        self._curves = []
        for n, i in enumerate(load_channels):
            curve = HDF5Plot()
            curve.setHDF5(array, i, y_spacing * n, scale=y_scale, dx=x_scale)
            curve.setClickable(True, width=10)
            curve.sigClicked.connect(self.highlight_curve)
            self.p1.addItem(curve)
            self.p1.addItem(curve.text)
            self._curves.append(curve)
        # this is an object to monitor (and manipulate?) the set of curves
        self.curve_collection = CurveCollection(self._curves)

        # Add mean trace to bottom plot
        self.nav_trace = pg.PlotCurveItem(x=np.arange(len(nav_trace)) * x_scale, y=nav_trace)
        self.p2.addItem(self.nav_trace)
        self.p2.setXRange(0, min(5e4, len(nav_trace))*x_scale)
        self.p2.setYRange(*np.percentile(nav_trace, [1, 99]))
        
        # Set bidirectional plot interaction
        self.region.sigRegionChanged.connect(self.update_zoom_callback)
        self.p1.sigXRangeChanged.connect(self.update_region_callback)

        # Do navigation jumps (if shift key is down)
        self.p2.scene().sigMouseClicked.connect(self.jump_nav)
        self.region.setRegion([0, 5000*x_scale])

        # Do fine interaction in zoomed plot with vertical line
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.p1.addItem(self.vline)
        self.p1.scene().sigMouseMoved.connect(self.fine_nav)
        
        # final adjustments to rows
        self.win.centralWidget.layout.setRowStretchFactor(0, 0.5)
        self.win.centralWidget.layout.setRowStretchFactor(1, 5)
        self.win.centralWidget.layout.setRowStretchFactor(2, 2.5)

        # a callable frame filter may be set on this object to affect frame display
        self.frame_filter = None

    def current_frame(self):
        return self.region.getRegion()

    def current_data(self):
        return self.curve_collection.current_data()

    def highlight_curve(self, curve):
        n = self._curves.index(curve)
        i, j = self.chan_map.rlookup(n)
        curve.toggle_text(i, j)
        
    def jump_nav(self, evt):
        if QtGui.QApplication.keyboardModifiers() != QtCore.Qt.ShiftModifier:
            return
        pos = evt.scenePos()
        if not self.p2.sceneBoundingRect().contains(pos):
            return
        newX = self.p2.vb.mapSceneToView(pos).x()
        minX, maxX = self.region.getRegion()
        rng = 0.5 * (maxX - minX)
        self.update_region([newX - rng, newX + rng])

    def fine_nav(self, evt):
        if QtGui.QApplication.keyboardModifiers() != QtCore.Qt.ShiftModifier:
            return
        #pos = evt.scenePos()
        pos = evt
        if not self.p1.sceneBoundingRect().contains(pos):
            return
        x_loc = self.p1.vb.mapSceneToView(pos).x()
        self.vline.setPos(x_loc)
        if not hasattr(self._curves[0], 'y_visible'):
            return
        self.set_image_frame(x=x_loc, move_vline=False)

    def set_image_frame(self, x=None, frame_vec=(), move_vline=True):
        if not len(frame_vec):
            if x is None:
                # can't do anything!
                return
            idx = self._curves[0].x_visible.searchsorted(x)
            frame_vec = np.empty(len(self.chan_map), 'd')
            for i in range(len(self.chan_map)):
                frame_vec[i] = self._curves[i].y_visible[idx]

        frame = embed_frame(self.chan_map, frame_vec)
        if self.frame_filter:
            frame = self.frame_filter(frame)
        self.img.setImage(frame, autoLevels=False)
        self.frame_text.setText('Time {:.3f}s'.format(x))
        if move_vline and x is not None:
            self.vline.setPos(x)
        
    def set_mean_image(self):
        if not hasattr(self._curves[0], 'y_visible'):
            return
        image = np.empty(len(self.chan_map), 'd')
        for i in range(len(self.chan_map)):
            image[i] = self._curves[i].y_visible.std()
        x_vis = self._curves[0].x_visible
        x_avg = 0.5 * (x_vis[0] + x_vis[-1])
        frame = embed_frame(self.chan_map, image)
        self.img.setImage(frame, autoLevels=False)
        self.frame_text.setText('Time ~ {:.3f}s'.format(x_avg))

    def update_zoom_from_region(self):
        # self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.p1.setXRange(minX, maxX, padding=0)
        # update image with mean of the current view interval
        self.set_mean_image()

    def update_zoom_callback(self):
        # in callback mode, disable the signal from the zoom plot
        with block_signals(self.p1):
            print('zoom callback called with block:', self.p1.signalsBlocked(), timestamp())
            self.update_zoom_from_region()

    def update_region(self, view_range):
        # region.setRegion appears to signal twice (once per side?),
        # so do this in two stages to prevent double signaling.
        orig_range = self.region.getRegion()
        # possibilities for region morphing:
        # 1) sliding / expanding right: right side increases, left side may or may not increase
        # 2) sliding / expanding left: left side decreases, right side may or may not decrease
        # 3) expanding in place: left side decreases and right side increases
        # 4) shrinking in place: left side increases and right side decreases
        #
        # (1), (3) and (4) move right side first
        # (2) move left side first: condition is that new_left < old_left and new_right <= old_right
        if view_range[0] < orig_range[0] and view_range[1] <= orig_range[1]:
            first_region = [view_range[0], orig_range[1]]
        else:
            first_region = [orig_range[0], view_range[1]]
        # do right-side while blocked (force block in case we're already in a blocking context)
        with block_signals(self.region, forced_state=True):
            print('setting range with block:', self.region.signalsBlocked(), timestamp())
            self.region.setRegion(first_region)
        # do both sides while emitting (or not)
        print('setting both sides with block:', self.region.signalsBlocked(), timestamp())
        self.region.setRegion(view_range)

    def update_region_callback(self, window, view_range):
        with block_signals(self.region):
            print('region callback called with block:', self.region.signalsBlocked(), timestamp())
            self.update_region(view_range)

    def update_y_spacing(self, offset):
        for i in range(len(self._curves)):
            self._curves[i].offset = i * offset
        x1, x2 = self.current_frame()
        # need to trigger a redraw with the new spacings--this should emit a data changed signal
        self.curve_collection.redraw_data(ignore_signals=False)
