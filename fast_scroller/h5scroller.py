from contextlib import contextmanager
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
pg.setConfigOptions(imageAxisOrder='row-major')
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecogdata.parallel.mproc import parallel_context, timestamp

from .helpers import DebounceCallback, CurveManager
from .curve_collections import PlotCurveCollection, LabeledCurveCollection


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
        self.win = pg.GraphicsLayoutWidget(border=(10, 10, 10))
        layout = self.win.ci
        # Adding columns to layout: just titles on the top row
        layout.addLabel('Array heatmap')
        layout.addLabel('Zoomed plot')
        layout.addLabel('|')
        # Next row has 1) the heatmap image with the colorbar widget
        layout.nextRow()
        sub_layout = layout.addLayout(colspan=1)
        self.img = pg.ImageItem(image=np.random.randn(*self.chan_map.geometry) * y_spacing)  # * 1e6 / 2)
        cmap = pg.colormap.get('coolwarm', source='matplotlib')
        p_img = sub_layout.addPlot()
        p_img.getViewBox().setAspectLocked()
        p_img.addItem(self.img)
        p_img.hideAxis('bottom')
        p_img.hideAxis('left')
        mid_x, top_y = self.chan_map.geometry[1] / 2.0, self.chan_map.geometry[0] + 2.0

        # add a text label on top of the box ("anchor" has text box is centered on its x, y position)
        self.frame_text = pg.TextItem('empty', anchor=(0.5, 0.5), color=(255, 255, 255))
        self.frame_text.setPos(mid_x, top_y)
        # self.vb_img.addItem(self.frame_text)
        p_img.getViewBox().addItem(self.frame_text)
        p_img.getViewBox().autoRange()

        # colorbar
        self.cb = pg.ColorBarItem(limits=None, cmap=cmap, hoverBrush='#EEEEFF80',
                                  rounding=10e-6, values=(-y_spacing, y_spacing))
        # forcibly reset image to a column (bug #1733 pyqtgraph)
        self.cb.bar.setImage(np.linspace(0, 1, 256)[:, np.newaxis])
        self.cb.getAxis('left').setLabel('')
        self.cb.getAxis('right').setLabel('Voltage', units='V')
        self.cb.setImageItem(self.img)
        sub_layout.addItem(self.cb)

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
        self.curve_manager = CurveManager(plot=self.p1)
        curves = PlotCurveCollection(array, load_channels, x_scale, y_scale, y_spacing)
        curves.setPen(color='w', width=1)
        self.curve_manager.add_new_curves(curves, 'all', set_source=True)
        # Set the heatmap to track these curves
        self.curve_manager.heatmap_name = 'all'

        # Selected curve & label set that calls up data on-demand
        labels = ['({}, {})'.format(i, j) for i, j in zip(*chan_map.to_mat())]
        selected_curves = LabeledCurveCollection(curves, labels, clickable=True)
        self.curve_manager.add_new_curves(selected_curves, 'selected')
        for text in selected_curves.texts:
            self.p1.addItem(text)

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
        
        # final adjustments to rows: args are (row, stretch)
        self.win.centralWidget.layout.setRowStretchFactor(0, 0.5)
        self.win.centralWidget.layout.setRowStretchFactor(1, 5)
        self.win.centralWidget.layout.setRowStretchFactor(2, 2.5)

        # a callable frame filter may be set on this object to affect frame display
        self.frame_filter = None

        # set up initial frame
        self.set_mean_image()

    def current_frame(self):
        return self.region.getRegion()

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
        if self.curve_manager.heatmap_curve.y_visible is None:
            return
        self.set_image_frame(x=x_loc, move_vline=False)

    def set_image_frame(self, x=None, frame_vec=(), move_vline=True):
        if not len(frame_vec):
            if x is None:
                # can't do anything!
                return
            idx = self.curve_manager.heatmap_curve.x_visible[0].searchsorted(x)
            frame_vec = self.curve_manager.heatmap_curve.y_visible[:, idx]
        chan_map = self.curve_manager.heatmap_curve.map_curves(self.chan_map)
        frame = embed_frame(chan_map, frame_vec)
        if self.frame_filter:
            frame = self.frame_filter(frame)
        info('Setting voltage frame {}'.format(timestamp()))
        # self.img.setImage(frame, autoLevels=False)
        self.img.setImage(frame, autoLevels=False)
        self.frame_text.setText('Time {:.3f}s'.format(x))
        if move_vline and x is not None:
            self.vline.setPos(x)
        
    def set_mean_image(self):
        if self.curve_manager.heatmap_curve.y_visible is None:
            return
        image = self.curve_manager.heatmap_curve.y_visible.std(axis=1)
        chan_map = self.curve_manager.heatmap_curve.map_curves(self.chan_map)
        x_vis = self.curve_manager.heatmap_curve.x_visible[0]
        x_avg = 0.5 * (x_vis[0] + x_vis[-1])
        frame = embed_frame(chan_map, image)
        info('setting RMS frame {}'.format(timestamp()))
        # self.img.setImage(frame, autoLevels=False)
        self.img.setImage(frame, autoLevels=False)
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
        self.curve_manager.source_curve.y_offset = offset
        # self.selected_curve_collection.y_offset = offset
