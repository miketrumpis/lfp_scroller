import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from ecogdata.util import ToggleState
from ecogdata.channel_map import ChannelMap
from ecogdata.parallel.mproc import parallel_context, timestamp

from .h5data import ReadCache
from . import pyqtSignal


info = parallel_context.get_logger().info

class PlotCurveCollection(pg.PlotCurveItem):
    data_changed = pyqtSignal(QtCore.QObject)
    plot_changed = pyqtSignal(QtCore.QObject)
    vis_rate_changed = pyqtSignal(float)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, data: ReadCache, plot_channels: list,
                 dx: float, dy:float, y_offset: float, *args, **kwargs):
        """
        This object is a PlotCurveItem that can draw a lot of disconnected curves that trace multi-channel signals.
        The data for the active page is drawn from a ReadCache source (i.e. HDF5). Under the hood there is a bit of
        machinery to avoid making unnecessary disk reads.

        The client-facing parts of this class provide:

        * current_data() -- "visible" data on the page (with or without the present trace offsets "y_offset")
        * set_external_data() -- setting the data on the page with filtered signals
        * event signals for: data changed, plot changed, on-screen sampling rate changed

        Parameters
        ----------
        data : ReadCache
            A buffer object that allows slicing into a file mapped 2D array
        plot_channels : list
            Which of the mapped array channels are to be plotted (and in which order)
        dx : float
            X-axis data rate (or reciprocal of signal sampling rate)
        dy : float
            Y-axis quantization: the Volts / data-unit conversion.
        y_offset : float
            Offset traces by this gap (Volts units)

        """
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
        y = self.y_visible.copy() if visible else self.y_slice.copy()
        # TODO: so x.shape isn't exactly consistent with y.shape if visible is False
        # TODO: option for raw data even if self.y_slice would return external (it would need to be sub-sliced to
        #  visible range)
        x = self.x_visible.copy() if full_xdata else self.x_visible[0].copy()
        return x, y

    # For consistency with subclasses, but all mapped channels are visible so it passes through
    def map_curves(self, channel_map: ChannelMap):
        return channel_map

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
            info('Zoom plot ranged actually changed, redrawing: {} {}'.format(self._view_range(), timestamp()))
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
            info('Slicing from HDF5 {} and emitting data_changed'.format(timestamp()))
            N = stop - start
            L = self.hdf_data.shape[1]
            # load this many points before and after the view limits so that small updates
            # can grab pre-loaded data
            extra = int(self.load_extra / 2 * N)
            sl = np.s_[max(0, start - extra):min(L, stop + extra)]
            self._use_raw_slice = True
            self._raw_slice = self.hdf_data[(self.plot_channels, sl)] * self.dy
            # self._raw_slice = self.hdf5[sl] * self._yscale
            self._cslice = sl
            self.data_changed.emit(self)

        # can retreive data within the pre-load
        x0 = self._cslice.start
        y_start = start - x0
        y_stop = y_start + (stop - start)
        self.y_visible = self.y_slice[:, y_start:y_stop]
        self.x_visible = np.tile(np.arange(start, stop) * self.dx, (len(self.plot_channels), 1))
        info('{}: x,y arrays set: {} {}'.format(type(self), self.y_visible.shape, self.x_visible.shape))

    def updatePlotData(self, data_ready=False):
        # Use data_ready=True to indicate that there's pre-defined x- and y-visible data ready to plot
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

        info('{}: Update hdf5 lines called: external data {} {}'.format(type(self), data_ready, timestamp()))
        visible = visible + self.y_offset
        connects = np.ones(visible.shape, dtype='>i4')
        connects[:, -1] = 0
        # Set the data as one block without connections from row to row
        info('{}: x shape {}, y shape {}, mask shape {}'.format(type(self),
                                                                x_visible.shape,
                                                                visible.shape,
                                                                connects.shape))
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
            self.vis_rate_changed.emit(float(vis_rate))
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


class FollowerCollection(PlotCurveCollection):
    can_update = ToggleState(init_state=False)
    connection_ids = list()

    def __new__(cls, curve_collection: PlotCurveCollection, *args, **kwargs):
        # Create a new sub-type of PlotCurveCollection and then initiate it with the source material
        obj = PlotCurveCollection.__new__(cls, curve_collection.hdf_data, curve_collection.plot_channels,
                                          curve_collection.dx, curve_collection.dy, curve_collection._y_offset)
                                          # *args, **kwargs)
        super(FollowerCollection, obj).__init__(curve_collection.hdf_data, curve_collection.plot_channels,
                                                curve_collection.dx, curve_collection.dy, curve_collection._y_offset)
                                                # *args, **kwargs)
        return obj

    def __init__(self, curve_collection: PlotCurveCollection):
        """
        This class works like a PlotCurveCollection, but only draws data from a source collection,
        and not from a raw data source. Whenever the source data is modified (e.g. by external filtering),
        those modifications are then applied to any follower collections.

        x-/y-visible data are retrieved by Property getters (setters do nothing), and y_offset is yoked to the source
        object. updatePlotData is gated by a context based ToggleState ("can_update").

        This class is essentially an abstract class that provides the curve-copying constructor and the gating
        mechanisms.

        Parameters
        ----------
        curve_collection : PlotCurveCollection
            Curve collection to clone into this follower.

        """
        self._source = curve_collection
        # TODO: do not think this is needed to follow source?
        # self.register_connection(curve_collection.data_changed, self._source_triggered_update)
        self.register_connection(curve_collection.plot_changed, self._source_triggered_update)
        self.register_connection(curve_collection.plot_changed, self._yoke_offset)
        self.setData([])

    @property
    def x_visible(self):
        return self._source.x_visible

    @x_visible.setter
    def x_visible(self, new):
        return

    @property
    def y_visible(self):
        return self._source.y_visible

    @y_visible.setter
    def y_visible(self, new):
        return

    def register_connection(self, signal: pyqtSignal, callback: callable):
        # register a signal connection so that it can be disabled later
        signal.connect(callback)
        self.connection_ids.append((signal, callback))

    def _cleanup(self):
        for signal, cb in self.connection_ids:
            try:
                signal.disconnect(cb)
            except TypeError:
                pass

    def _yoke_offset(self, source):
        if source._y_offset != self._y_offset:
            with self.can_update(True):
                self.y_offset = source._y_offset

    def _source_triggered_update(self, source):
        blocking = source._y_offset != self._y_offset
        # Proceed to update unless this has been triggered by an offset change.
        # The yoke_offset callback will handle that
        with self.can_update(not blocking):
            self.updatePlotData()

    # This method will be triggered by the PlotItem. Ignore it if the source data is not ready.
    # This method is also attached indirectly to data_changed and plot_changed signals.
    def viewRangeChanged(self):
        return

    def updatePlotData(self, data_ready=False):
        # By default, plots everything that is visible on the source.
        if not self.can_update:
            return
        # info('updating based on source data: ', self.x_visible.shape, self.y_visible.shape)
        super().updatePlotData(data_ready=True)


class SelectedFollowerCollection(FollowerCollection):

    _available_channels: list
    _active_channels: np.ndarray
    selection_changed = pyqtSignal(np.ndarray)

    def __init__(self, curve_collection: PlotCurveCollection, clickable: bool=False, init_active: bool=False,
                 pen_args=None, shadowpen_args=None):
        """
        This kind of FollowerCollection modifies its visible data (through Property logic) to show only a subset of the
        source channels.

        Parameters
        ----------
        curve_collection : PlotCurveCollection
            Source collection to follow
        clickable : bool
            Curves can be selected by mouse click
        init_active : bool
            Channels begin active (True) or inactive (False)
        pen_args : dict
            For constructing the Pen that draws the selected curves
        shadowpen_args : dict
            For constructing the ShadowPen that highlights the selected curves

        """
        super().__init__(curve_collection)
        if pen_args is None:
            pen_args = dict(width=0)
        if shadowpen_args is None:
            shadowpen_args = dict(width=4, color='c')
        self._available_channels = self._source.plot_channels[:]
        self._active_channels = np.zeros(len(self._available_channels), dtype='?')
        if init_active:
            self._active_channels[:] = True
        self.setPen(**pen_args)
        self.setShadowPen(**shadowpen_args)
        if clickable:
            curve_collection.setClickable(True, width=10)
            curve_collection.sigClicked.connect(self.report_clicked)

    def report_clicked(self, curves: PlotCurveCollection, event: QtCore.QEvent):
        ex, ey = event.pos()
        vx, vy = curves.current_data()
        vy = vy[:, vx.searchsorted(ex)] + curves.y_offset.squeeze()
        channel = np.argmin(np.abs(vy - ey))
        b = self._active_channels[channel]
        print('click pos', event.pos(), 'channel', channel, 'changing to', not b)
        self._active_channels[channel] = not b
        with self.can_update(True):
            self.updatePlotData()
        self.selection_changed.emit(self._active_channels)

    def set_selection(self, selected=None):
        """
        Set the selection to this subset.

        Parameters
        ----------
        selected : array-like
            Subset of plot channels to select. If None, then selection is blank.

        """
        self._active_channels[:] = False
        if selected is not None:
            self._active_channels[selected] = True
        with self.can_update(True):
            self.updatePlotData()
        self.selection_changed.emit(self._active_channels)

    def map_curves(self, channel_map: ChannelMap):
        return channel_map.subset(self._active_channels)

    @property
    def x_visible(self):
        if not self._active_channels.any():
            return None
        x = self._source.x_visible
        if x is not None:
            x = x[self._active_channels]
        return x

    @x_visible.setter
    def x_visible(self, new):
        return

    @property
    def y_visible(self):
        if not self._active_channels.any():
            return None
        y = self._source.y_visible
        if y is not None:
            y = y[self._active_channels]
        return y

    @y_visible.setter
    def y_visible(self, new):
        return

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
        with self.can_update(True):
            self.updatePlotData()


class LabeledCurveCollection(SelectedFollowerCollection):

    def __init__(self, curve_collection: PlotCurveCollection, label_set: list,
                 clickable: bool=False, pen_args=None, shadowpen_args=None):
        """
        A SelectedFollowerCollection that also has labels for the selected curves.

        Parameters
        ----------
        curve_collection : PlotCurveCollection
            Source collection to follow
        label_set: list
            Label for each source curve
        clickable : bool
            Curves can be selected by mouse click
        pen_args : dict
            For constructing the Pen that draws the selected curves
        shadowpen_args : dict
            For constructing the ShadowPen that highlights the selected curves

        """
        super().__init__(curve_collection, clickable=clickable, pen_args=pen_args, shadowpen_args=shadowpen_args)
        self.texts = list()
        for label in label_set:
            text = pg.TextItem(label, anchor=(0.5, 1.0), color=(255, 98, 65), fill=(65, 222, 255, 180))
            text.setVisible(False)
            # text.setText(label)
            self.texts.append(text)
        self.selection_changed.connect(self.set_text_position)
        self.register_connection(curve_collection.plot_changed, self.set_text_position)

    def set_text_position(self, selection):
        # the number of rows in x- and y-visible is equal to the number of active channels
        x_visible = self.x_visible
        y_visible = self.y_visible
        if y_visible is not None:
            y_visible = y_visible + self.y_offset
        label_row = 0
        for i, text in enumerate(self.texts):
            text.setVisible(self._active_channels[i])
            if not text.isVisible():
                continue
            # put text 20% from the left
            x_pos = 0.8 * x_visible[label_row, 0] + 0.2 * x_visible[label_row, -1]
            # set height to the max in a 10% window around x_pos
            wid = max(1, int(0.05 * y_visible.shape[1]))
            n_pos = int(0.2 * y_visible.shape[1])
            y_pos = y_visible[label_row, n_pos - wid:n_pos + wid].max()
            info('Setting text {}, {}: {}'.format(x_pos, y_pos, timestamp()))
            # print('Setting text {}, {}: {}'.format(x_pos, y_pos, timestamp()))
            text.setPos(x_pos, y_pos)
            label_row += 1

    @property
    def labels(self):
        return [text.getText() for text in self.texts if text.isVisible()]


