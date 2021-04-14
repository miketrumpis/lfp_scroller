from contextlib import contextmanager
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
# from .pyqtgraph_extensions import ImageItem, ColorBarItem, get_colormap_lut
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecogdata.parallel.mproc import parallel_context, timestamp
from time import time

from .helpers import DebounceCallback
from .h5data import ReadCache


info = parallel_context.get_logger().info


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
    qtobj: (list of) QtCore.QObject
    forced_state: bool or None

    """
    if not isinstance(qtobj, (list, tuple)):
        qtobj = [qtobj]
    n_objs = len(qtobj)
    if forced_state is not None:
        states = [not forced_state] * n_objs
    else:
        states = [qo.signalsBlocked() for qo in qtobj]
    try:
        orig_state = [qo.blockSignals(not state) for (qo, state) in zip(qtobj, states)]
        yield
    finally:
        for qo, state in zip(qtobj, orig_state):
            qo.blockSignals(state)


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


class PlotCurveCollection(pg.PlotCurveItem):
    data_changed = QtCore.pyqtSignal(QtCore.QObject)
    plot_changed = QtCore.pyqtSignal(QtCore.QObject)
    vis_rate_changed = QtCore.pyqtSignal(np.float64)

    def __init__(self, data: ReadCache, plot_channels: list,
                 dx: float, dy:float, y_offset: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These are HDF5 getter helpers per plot curve line
        self.hdf_data = data
        self._y_offset = y_offset
        self.dx = dx
        self.dy = dy
        self.plot_channels = plot_channels
        # maximum number of samples to be plotted
        self.limit = kwargs.pop('points_limit', 6000)
        self.load_extra = kwargs.pop('load_extra', 0.25)
        self._current_vis_rate = 1 / dx
        # This item also has a curve point and (in)visible text
        # self.curve_point = pg.CurvePoint(self)
        # self.text = pg.TextItem('empty', anchor=(0.5, 1.0), color=(255, 98, 65), fill=(65, 222, 255, 180))
        # self.text.setParent(self.curve_point)
        # self.text.setVisible(False)
        # track current values of (x1, x2) to protect against reloads triggered by spurious view changes
        self._cx1 = 0
        self._cx2 = 0
        # track current ACTUAL slice (including any pre-loaded buffer)
        self._cslice = None
        # Set some state tracking to know whether to use slices from the underlying HDF buffer,
        # or modified/filtered versions set by an external module. This state only affects
        # the underlying loaded data, which is a superset of the visible data
        self._use_raw_slice = True
        self._raw_slice = None
        self._external_slice = None
        self.x_visible = None
        self.y_visible = None

    @property
    def y_offset(self):
        return (np.arange(len(self.plot_channels)) * self._y_offset)[:, None]

    @y_offset.setter
    def y_offset(self, new):
        self._y_offset = new
        self.updatePlotData()

    def current_data(self, visible=True, full_xdata=False):
        """
        Returns an array of the available display data without offsets

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
        if self.y_visible is None:
            return None, None
        y = self.y_visible if visible else self.y_slice
        # TODO: so x.shape isn't exactly consistent with y.shape if visible is False
        x = self.x_visible if full_xdata else self.x_visible[0]
        return x, y

    @property
    def y_slice(self):
        return self._raw_slice if self._use_raw_slice else self._external_slice

    def _view_really_changed(self):
        r = self._view_range()
        if r is None:
            return False
        return (r[0] != self._cx1) or (r[1] != self._cx2)

    def viewRangeChanged(self):
        # only look for x-range changes that actually require loading different data
        if self._view_really_changed():
            info('Zoom plot ranged actually changed, '
                 'triggering data load: {} {}'.format(self._view_range(), timestamp()))
            self.updatePlotData()

    def _view_range(self):
        vb = self.parentWidget()
        if vb is None:
            return  # no ViewBox yet
        # Determine what data range must be read from HDF5
        xrange = vb.viewRange()[0]
        srange = list(map(lambda x: int(x / self.dx), xrange))
        start = max(0, int(srange[0]) - 1)
        stop = min(self.hdf_data.shape[-1], int(srange[1] + 2))
        return start, stop

    def _set_data(self):
        r = self._view_range()
        if not r:
            self.y_visible = None
            self.x_visible = None
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
            # When this condition, automatically reset state to raw slices.
            # If anything is watching this data to change, it will need access to the raw slice.
            info('Slicing from HDF5 {}'.format(timestamp()))
            N = stop - start
            L = self.hdf_data.shape[1]
            # load this many points before and after the view limits so that small updates
            # can grab pre-loaded data
            extra = int(self.load_extra / 2 * N)
            sl = np.s_[max(0, start - extra):min(L, stop + extra)]
            self._use_raw_slice = True
            # TODO: note this should now be setting a FULL array of raw slice on this object (no need for individual
            #  raw slices)
            self._raw_slice = self.hdf_data[(self.plot_channels, sl)] * self.dy
            # self._raw_slice = self.hdf5[sl] * self._yscale
            self._cslice = sl
            self.data_changed.emit(self)

        # can retreive data within the pre-load
        x0 = self._cslice.start
        y_start = start - x0
        y_stop = y_start + (stop - start)
        # TODO: this is calling up whatever existing data in the "y_slice" property
        self.y_visible = self.y_slice[:, y_start:y_stop]
        self.x_visible = np.tile(np.arange(start, stop) * self.dx, (len(self.plot_channels), 1))
        print('x,y arrays set:', self.y_visible.shape, self.x_visible.shape)

    def updatePlotData(self, data_ready=False):
        if self.hdf_data is None or not len(self.plot_channels):
            self.setData([])
            self.y_visible = None
            self.x_visible = None
            return

        if not data_ready:
            self._set_data()
        r = self._view_range()
        if not r:
            return
        start, stop = r
        # Decide by how much we should downsample
        ds = int((stop - start) / self.limit) + 1
        if ds == 1:
            # Small enough to display with no intervention.
            visible = self.y_visible
            x_visible = self.x_visible
        else:
            # Here convert data into a down-sampled
            # array suitable for visualizing.

            N = self.y_visible.shape[1]

            # This downsample does min/max interleaving
            # -- does not squash noise
            # 1st axis: number of display points
            # 2nd axis: number of raw points per display point
            # ds = ds * 2
            # Nsub = max(5, N // ds)
            # y = self.y_visible[:Nsub * ds].reshape(Nsub, ds)
            # visible = np.empty(Nsub * 2, dtype='d')
            # visible[0::2] = y.min(axis=1)
            # visible[1::2] = y.max(axis=1)

            # This downsample picks 2 points regularly spaced in the segment
            ds = ds * 2
            Nsub = max(5, N // ds)
            p1 = ds // 4
            p2 = (3 * ds) // 4
            n_rows = self.y_visible.shape[0]
            y = self.y_visible[:, :Nsub * ds].reshape(n_rows, Nsub, ds)
            visible = np.empty((n_rows, Nsub * 2), dtype='d')
            visible[:, 0::2] = y[..., p1]
            visible[:, 1::2] = y[..., p2]

            # This downsample does block averaging
            # -- this flattens noisy segments
            # Nsub = max(5, N // ds)
            # y = self.y_visible[:Nsub * ds].reshape(Nsub, ds)
            # visible = y.mean(axis=1)

            x_samp = np.linspace(ds // 2, self.x_visible.shape[1] - ds // 2, Nsub * 2).astype('i')
            if x_samp[-1] >= self.x_visible.shape[1]:
                x_samp[-1] = N - 1
            x_visible = np.take(self.x_visible, x_samp, axis=1)

        info('Update hdf5 lines called: external data {} {}'.format(data_ready, timestamp()))
        visible = visible + self.y_offset
        connects = np.ones(visible.shape, dtype='>i4')
        connects[:, -1] = 0
        # Set the data as one block without connections from row to row
        info('x shape {}, y shape {}, mask shape {}'.format(x_visible.shape, visible.shape, connects.shape))
        self.setData(x=x_visible.ravel(), y=visible.ravel(), connect=connects.ravel())  # update_zoom_callback the plot
        # TODO: is this needed?
        self.resetTransform()
        # self.set_text_position()
        # mark these start, stop positions to match against future view-change signals
        self._cx1 = start
        self._cx2 = stop
        vis_rate = round(1 / (x_visible[0, 1] - x_visible[0, 0]))
        if vis_rate != self._current_vis_rate:
            self._current_vis_rate = vis_rate
            self.vis_rate_changed.emit(vis_rate)
        self.plot_changed.emit(self)

    def set_external_data(self, array, redraw=False, visible=False):
        # visible == True only changes the on-screen data, rather than the extended slice
        if visible:
            if array.shape[1] != self.x_visible.shape[1]:
                raise ValueError('Shape of external series does not match the current window.')
            self.y_visible = array
            if self.x_visible is None:
                self.x_visible = np.tile(np.arange(array.shape[1]), (array.shape[0], 1))
            if redraw:
                self.updatePlotData(data_ready=True)
        else:
            if array.shape[1] != (self._cslice.stop - self._cslice.start):
                raise ValueError('Shape of external series does not match the cached load.')
            # If setting on the underlying slice, then set it here and update the mode
            self._external_slice = array
            self._use_raw_slice = False
            # update as if data is cached
            if redraw:
                self.updatePlotData()

    def unset_external_data(self, redraw=False, visible=False):
        info('Unsetting external (visible {}) and updating curve data {}'.format(visible, timestamp()))
        if not visible:
            # Revert to raw slices and reset visible data
            self._use_raw_slice = True
        if redraw:
            self.updatePlotData()


class LabeledCurveCollection(PlotCurveCollection):

    def __init__(self, data: ReadCache, plot_channels: list, label_set: list,
                 dx: float, dy: float, y_offset: float, *args, **kwargs):

        super().__init__(data, list(), dx, dy, y_offset, *args, **kwargs)
        self.setPen(width=0)
        self.setShadowPen(color='c', width=4)
        self._available_channels = plot_channels[:]
        self._active_channels = np.zeros(len(plot_channels), dtype='?')

        # text item can be looked up by position (which is found through closest row in the visible data)
        self.texts = list()
        for label in label_set:
            text = pg.TextItem(label, anchor=(0.5, 1.0), color=(255, 98, 65), fill=(65, 222, 255, 180))
            text.setVisible(False)
            text.setText(label)
            self.texts.append(text)
        self.plot_changed.connect(self.set_text_position)

    @property
    def plot_channels(self):
        can_plot = [self._available_channels[i] for i in self.selected_channels]
        return can_plot

    # This is just here to allow the parent class to do self.plot_channels = foo
    @plot_channels.setter
    def plot_channels(self, val):
        return

    @property
    def selected_channels(self):
        return np.where(self._active_channels)[0]

    @property
    def y_offset(self):
        active = self.selected_channels
        if not len(active):
            return 0
        return (active * self._y_offset)[:, None]

    @y_offset.setter
    def y_offset(self, new):
        self._y_offset = new
        self.updatePlotData()

    def report_clicked(self, curves: PlotCurveCollection, event: QtCore.QEvent):
        ex, ey = event.pos()
        vx, vy = curves.current_data()
        vy = vy[:, vx.searchsorted(ex)] + curves.y_offset.squeeze()
        channel = np.argmin(np.abs(vy - ey))
        # print('click pos', event.pos(), 'channel', channel)
        b = self._active_channels[channel]
        self._active_channels[channel] = not b
        self.texts[channel].setVisible(not b)
        self._cslice = None  # This will trigger a data reload
        self.updatePlotData()


    def set_text_position(self):
        x_visible = self.x_visible
        y_visible = self.y_visible + self.y_offset
        i = 0
        for text in self.texts:
            if not text.isVisible():
                continue
            # put text 20% from the left
            x_pos = 0.8 * x_visible[i, 0] + 0.2 * x_visible[i, -1]
            # set height to the max in a 10% window around x_pos
            wid = max(1, int(0.05 * y_visible.shape[1]))
            n_pos = int(0.2 * y_visible.shape[1])
            y_pos = y_visible[i, n_pos - wid:n_pos + wid].max()
            info('Setting text {}, {}: {}'.format(x_pos, y_pos, timestamp()))
            print('Setting text {}, {}: {}'.format(x_pos, y_pos, timestamp()))
            text.setPos(x_pos, y_pos)
            i += 1


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
        if isinstance(load_channels, str) and load_channels == 'all':
            load_channels = list(range(len(array)))
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

        self.img = pg.ImageItem(image=np.random.randn(*self.chan_map.geometry) * y_spacing * 1e6 / 2)
        cmap = pg.colormap.get('coolwarm', source='matplotlib')
        self.p_img.addItem(self.img)
        self.p_img.autoRange()
        self.cb = pg.ColorBarItem(limits=None, width=25, cmap=cmap, hoverBrush='#EEEEFF80',
                                  rounding=10)
        # self.cb.enableAutoRange()
        self.cb.setLevels(low=-y_spacing * 1e6, high=y_spacing * 1e6)
        self.cb.setImageItem(self.img)
        self.cb.getAxis('left').setLabel('')
        self.cb.getAxis('right').setLabel('Voltage ({})'.format(units))
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

        # Multiple curve set that calls up data on-demand
        curves = PlotCurveCollection(array, load_channels, x_scale, y_scale, y_spacing)
        self.p1.addItem(curves)
        self.curve_collection = curves

        # Selected curve & label set that calls up data on-demand
        labels = ['({}, {})'.format(i, j) for i, j in zip(*chan_map.to_mat())]
        selected_curves = LabeledCurveCollection(array, load_channels, labels, x_scale, y_scale, y_spacing)
        self.p1.addItem(selected_curves)
        for text in selected_curves.texts:
            self.p1.addItem(text)
        self.selected_curve_collection = selected_curves
        self.curve_collection.setClickable(True, width=None)
        self.curve_collection.sigClicked.connect(self.selected_curve_collection.report_clicked)

        # Add mean trace to bottom plot
        self.nav_trace = pg.PlotCurveItem(x=np.arange(len(nav_trace)) * x_scale, y=nav_trace)
        self.p2.addItem(self.nav_trace)
        self.p2.setXRange(0, min(5e4, len(nav_trace))*x_scale)
        self.p2.setYRange(*np.percentile(nav_trace, [1, 99]))
        
        # Set bidirectional plot interaction
        # need to hang onto references?
        self._db_cnx1 = DebounceCallback.connect(self.region.sigRegionChanged, self.update_zoom_callback)
        self._db_cnx2 = DebounceCallback.connect(self.p1.sigXRangeChanged, self.update_region_callback)

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

        # set up initial frame
        self.set_mean_image()

    def current_frame(self):
        return self.region.getRegion()

    def current_data(self):
        return self.curve_collection.current_data()

    # def highlight_curve(self, curve):
    #     n = self._curves.index(curve)
    #     i, j = self.chan_map.rlookup(n)
    #     curve.toggle_text(i, j)
        
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
        if self.curve_collection.y_visible is None:
            return
        self.set_image_frame(x=x_loc, move_vline=False)

    def set_image_frame(self, x=None, frame_vec=(), move_vline=True):
        if not len(frame_vec):
            if x is None:
                # can't do anything!
                return
            idx = self.curve_collection.x_visible[0].searchsorted(x)
            frame_vec = self.curve_collection.y_visible[:, idx]

        frame = embed_frame(self.chan_map, frame_vec)
        if self.frame_filter:
            frame = self.frame_filter(frame)
        info('Setting voltage frame {}'.format(timestamp()))
        # self.img.setImage(frame, autoLevels=False)
        self.img.setImage(frame * 1e6, autoLevels=False)
        self.frame_text.setText('Time {:.3f}s'.format(x))
        if move_vline and x is not None:
            self.vline.setPos(x)
        
    def set_mean_image(self):
        if self.curve_collection.y_visible is None:
            return
        image = self.curve_collection.y_visible.std(axis=1)
        x_vis = self.curve_collection.x_visible[0]
        x_avg = 0.5 * (x_vis[0] + x_vis[-1])
        frame = embed_frame(self.chan_map, image)
        info('setting RMS frame {}'.format(timestamp()))
        # self.img.setImage(frame, autoLevels=False)
        self.img.setImage(frame * 1e6, autoLevels=False)
        self.frame_text.setText('Time ~ {:.3f}s'.format(x_avg))

    def update_zoom_from_region(self):
        # self.region.setZValue(10)
        minX, maxX = self.region.getRegion()
        self.p1.setXRange(minX, maxX, padding=0)
        # update image with mean of the current view interval
        self.set_mean_image()

    def update_zoom_callback(self, *args):
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
        info = parallel_context.get_logger().info
        with block_signals(self.region, forced_state=True):
            info('setting first range with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
            self.region.setRegion(first_region)
        # do both sides while emitting (or not)
        info('setting both sides with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
        self.region.setRegion(view_range)

    def update_region_callback(self, window, view_range, *args):
        info = parallel_context.get_logger().info
        with block_signals(self.region):
            info('region callback called with block: {} {}'.format(self.region.signalsBlocked(), timestamp()))
            self.update_region(view_range)

    def update_y_spacing(self, offset):
        self.curve_collection.y_offset = offset
        self.selected_curve_collection.y_offset = offset
        # for i in range(len(self._curves)):
        #     self._curves[i].offset = i * offset
        # x1, x2 = self.current_frame()
        # # need to trigger a redraw with the new spacings--this should emit a data changed signal
        # self.curve_collection.redraw_data_offsets(ignore_signals=False)
