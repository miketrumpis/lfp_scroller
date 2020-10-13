from contextlib import contextmanager
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
from .pyqtgraph_extensions import ImageItem, ColorBarItem, get_colormap_lut
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecogdata.parallel.array_split import timestamp
from ecogdata.parallel.mproc import get_logger
from time import time


info = get_logger().info


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


class HDF5Plot(pg.PlotCurveItem):
    def __init__(self, *args, **kwds):
        self.hdf5 = None
        # maximum number of samples to be plotted
        self.limit = kwds.pop('points_limit', 6000)
        self.load_extra = kwds.pop('load_extra', 0.25)
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
        # track current ACTUAL slice (including any pre-loaded buffer)
        self._cslice = None
        
    def setHDF5(self, data, index, offset, number, dx=1.0, scale=1):
        self.hdf5 = data
        self.index = index
        self._yscale = scale
        self._xscale = dx
        self.offset = offset
        self.number = number
        self.updateHDF5Plot()

    @property
    def selected(self):
        return self.text.isVisible()

    def _view_really_changed(self):
        r = self._view_range()
        if r is None:
            return False
        return (r[0] != self._cx1) or (r[1] != self._cx2)

    def viewRangeChanged(self):
        # only look for x-range changes that actually require loading different data
        if self._view_really_changed():
            if self.offset == 0:
                info('Zoom plot ranged actually changed, '
                     'triggering data load: {} {}'.format(self._view_range(), timestamp()))
            self.updateHDF5Plot()

    def _view_range(self):
        vb = self.parentWidget()
        if vb is None:
            return  # no ViewBox yet
        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        srange = list(map(lambda x: int(x / self._xscale), xrange))
        start = max(0, int(srange[0]) - 1)
        stop = min(self.hdf5.shape[-1], int(srange[1] + 2))
        return start, stop
        
    def _set_data(self):
        r = self._view_range()
        if not r:
            return
        start, stop = r

        if self._cslice is None:
            needs_slice = True
        else:
            cstart = self._cslice.start
            cstop = self._cslice.stop
            # If the any view limit is outside the old slice bounds, update the loaded data
            needs_slice = start < cstart or stop > cstop

        if needs_slice:
            if self.offset == 0:
                info('Slicing from HDF5 {}'.format(timestamp()))
            N = stop - start
            L = self.hdf5.shape[1]
            # load this many points before and after the view limits so that small updates
            # can grab pre-loaded data
            extra = int(self.load_extra / 2 * N)
            sl = np.s_[self.index, max(0, start - extra):min(L, stop + extra)]
            self._y_loaded = self.hdf5[sl] * self._yscale
            self._cslice = sl[1]

        # can retreive data within the pre-load
        x0 = self._cslice.start
        y_start = start - x0
        y_stop = y_start + (stop - start)
        self.y_visible = self._y_loaded[y_start:y_stop]
        self.x_visible = np.arange(start, stop) * self._xscale

    def set_external_data(self, y, visible=False):
        if visible:
            if len(y) != len(self.x_visible):
                raise ValueError('Shape of external series does not match the current window.')
            self.y_visible = y
            self.updateHDF5Plot(data_ready=True)
        else:
            if len(y) != (self._cslice.stop - self._cslice.start):
                raise ValueError('Shape of external series does not match the cached load.')
            self._y_loaded = y
            # update as if data is cached
            self.updateHDF5Plot()

    def unset_external_data(self, visible=False):
        if self.offset == 0:
            info('Unsetting external (visible {}) and updating curve data {}'.format(visible, timestamp()))
        if not visible:
            # Clear the saved slice to trigger a reload
            self._cslice = None
        self.updateHDF5Plot()

    def update_visible_offset(self):
        if self.parentWidget() is None:
            return
        tic = time()
        y_visible = self.y_visible + self.offset
        self.setData(x=self.x_visible, y=y_visible)
        # self.resetTransform()
        # self.set_text_position()
        toc = time() - tic
        info = get_logger().info
        info('set data time {}'.format(toc))
        
    def updateHDF5Plot(self, data_ready=False):
        # if self.offset == 0:
        info = get_logger().info
        info('Update hdf5 lines called num {}: external data {} {}'.format(self.number, data_ready, timestamp()))
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
            # 1st axis: number of display points
            # 2nd axis: number of raw points per display point
            y = self.y_visible[:Nsub * ds].reshape(Nsub, ds)

            # This downsample does min/max interleaving
            # visible = np.empty(Nsub * 2, dtype='d')
            # visible[0::2] = y.min(axis=1)
            # visible[1::2] = y.max(axis=1)

            # This downsample does block averaging
            visible = y.mean(axis=1)

            x_samp = np.linspace(ds // 2, len(self.x_visible) - ds // 2, len(visible)).astype('i')
            if x_samp[-1] >= len(self.x_visible):
                x_samp[-1] = N - 1
            x_visible = np.take(self.x_visible, x_samp)

        visible = visible + self.offset
        self.setData(x=x_visible, y=visible)  # update_zoom_callback the plot
        self.resetTransform()
        self.set_text_position()
        # mark these start, stop positions to match against future view-change signals
        self._cx1 = start
        self._cx2 = stop

    def set_text_position(self):
        if self.text.isVisible():
            x_visible = self.x_visible
            y_visible = self.y_visible
            # put text 20% from the left
            x_pos = 0.8 * x_visible[0] + 0.2 * x_visible[-1]
            # set height to the max in a 10% window around x_pos
            wid = max(1, int(0.05 * len(y_visible)))
            n_pos = int(0.2 * len(y_visible))
            y_pos = y_visible[n_pos - wid:n_pos + wid].max() + self.offset
            info('Setting text {}, {}: {}'.format(x_pos, y_pos, timestamp()))
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


class CurveCollection(QtCore.QObject):
    data_changed = QtCore.pyqtSignal(QtCore.QObject)
    plot_changed = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self, curves, *args, **kwargs):
        super(CurveCollection, self).__init__(*args, **kwargs)
        self.curves = curves
        self._current_set = set()
        # self._connections = list()
        self._current_slice = slice(None)
        for c in curves:
            c.sigPlotChanged.connect(self.count_curves)
            # self._connections.append(c.sigPlotChanged.connect(self.count_curves))

    def count_curves(self, curve):
        """
        This callback monitors the "plot changed" signal on the curve collection.
        When a curve emits, add it to a list.
        When the list is full, then emit the "plot changed" signal to update graphics.
        Separately, if the underlying curve data has changed, emit the "data changed" signal

        """
        self._current_set.add(curve)
        if len(self._current_set) == len(self.curves):
            self.plot_changed.emit(self)
            info('CurveCollection emitting plot_changed and resetting current set')
            self._current_set = set()
            sl = self.curves[0]._cslice
            if sl != self._current_slice:
                self.data_changed.emit(self)
                self._current_slice = sl
                info('CurveCollection emitting data_changed')
        else:
            info('added {} len curve set {}'.format(curve.number, len(self._current_set)))


    def selected_channels(self):
        return [i for i, c in enumerate(self.curves) if c.selected]

    def current_data(self, visible=True):
        """
        Returns an array of the available display data.

        Parameters
        ----------
        visible: bool
            By default, return the visibly plotted data. If False, return any pre-loaded data as well.
            In this case, the x array will not be aligned with y.

        Returns
        -------
        x: ndarray
        y: ndarray

        """
        x_vis = self.curves[0].x_visible
        y_vis = self.curves[0].y_visible if visible else self.curves[0]._y_loaded
        n_curves = len(self.curves)
        x = np.empty((n_curves, len(x_vis)), x_vis.dtype)
        y = np.empty((n_curves, len(y_vis)), y_vis.dtype)
        x[0] = x_vis
        y[0] = y_vis
        for i in range(1, n_curves):
            x[i] = self.curves[i].x_visible
            y[i] = self.curves[i].y_visible if visible else self.curves[i]._y_loaded
        return x, y

    @contextmanager
    def disable_listener(self, blocking=True):
        if blocking:
            info('Blocking sigPlotChanged callbacks {}'.format(timestamp()))
            for curve in self.curves:
                # just disconnect the slot (i.e. the callable)
                curve.sigPlotChanged.disconnect(self.count_curves)
        try:
            yield
        finally:
            if blocking:
                info('Reinstating sigPlotChanged callbacks {}'.format(timestamp()))
                for c in self.curves:
                    c.sigPlotChanged.connect(self.count_curves)

    def set_data_on_collection(self, array, ignore_signals=True, visible=True):
        info('Setting external data to curves: '
             'ignore {}, visible {} {}'.format(ignore_signals, visible, timestamp()))
        with self.disable_listener(blocking=ignore_signals):
            for data, line in zip(array, self.curves):
                line.set_external_data(data, visible=visible)

    def unset_data_on_collection(self, ignore_signals=True, visible=True):
        info('Unsetting external array on curves, blocking: {} {}'.format(ignore_signals, timestamp()))
        with self.disable_listener(blocking=ignore_signals):
            for line in self.curves:
                line.unset_external_data(visible=visible)

    def redraw_data(self, ignore_signals=False):
        info('calling updateHDF5Plot on curves, ignore {} {}'.format(ignore_signals, timestamp()))
        with self.disable_listener(blocking=ignore_signals):
            for line in self.curves:
                line.updateHDF5Plot()
        info('done updateHDF5Plot on curves, {}'.format(timestamp()))

    def redraw_data_offsets(self, ignore_signals=False):
        # this would be used e.g. when the y-offsets have changed but the view limits have not
        # block on all but the last curve, then do as instructed for the final curve
        info('start redrawing curve collection {}'.format(timestamp()))
        with self.disable_listener(blocking=ignore_signals):
            for line in self.curves:
                line.update_visible_offset()
        info('end redrawing curve collection {}'.format(timestamp()))


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

        self.p1.enableAutoRange(axis='y', enable=True)
        self.p1.setAutoVisible(y=False)
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
        initial_pts = 5000
        self.region.setRegion([0, initial_pts * x_scale])

        # Add the LinearRegionItem to the ViewBox,
        # but tell the ViewBox to exclude this 
        # item when doing auto-range calculations.
        self.p2.addItem(self.region, ignoreBounds=True)

        self.p1.setAutoVisible(y=True)
        self.p1.setXRange(0, initial_pts * x_scale)
        self.p1.vb.setLimits(maxXRange=max_zoom)

        # curves will be itemized in order with the channel map
        self._curves = []
        max_visible_points = 1e6 // len(load_channels)
        for n, i in enumerate(load_channels):
            curve = HDF5Plot(points_limit=max_visible_points)
            curve.setHDF5(array, i, y_spacing * n, n, scale=y_scale, dx=x_scale)
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
        info('Jump-updating region box {}'.format(timestamp()))
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
        info('Setting voltage frame {}'.format(timestamp()))
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
        info('setting RMS frame {}'.format(timestamp()))
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
            info('zoom callback called with block: {} {}'.format(self.p1.signalsBlocked(), timestamp()))
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
        info = get_logger().info
        with block_signals(self.region, forced_state=True):
            info('setting first range with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
            self.region.setRegion(first_region)
        # do both sides while emitting (or not)
        info('setting both sides with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
        self.region.setRegion(view_range)

    def update_region_callback(self, window, view_range):
        info = get_logger().info
        with block_signals(self.region):
            info('region callback called with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
            self.update_region(view_range)

    def update_y_spacing(self, offset):
        for i in range(len(self._curves)):
            self._curves[i].offset = i * offset
        x1, x2 = self.current_frame()
        # need to trigger a redraw with the new spacings--this should emit a data changed signal
        self.curve_collection.redraw_data_offsets(ignore_signals=False)
