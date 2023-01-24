from typing import Union
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from traits.api import HasTraits, Instance, Button, Int, Enum, Float, Str, Bool, Property, Tuple, observe
from traitsui.api import View, UItem, Handler, Group, HGroup, VGroup, Label
from pyqtgraph.Qt import QtCore

from ecoglib.estimation.multitaper import bw2nw
from ecogdata.util import nextpow2, ToggleState
from ecogdata.parallel.array_split import split_at
from ecogdata.filt.time import moving_projection, slepian_projection, filter_array

from .base import VisModule
from ..helpers import DebounceCallback, view_label_item
from ..curve_collections import SelectedFollowerCollection, PlotCurveCollection
# from .. import pyqtSignal

# Most of the classes defined here inherit from HasTraits from the traits package. "Traits" can be used
# programmatically like regular Python types (e.g. int, str, list). In addition, a trait can also signal when its
# value changes to downstream listeners, and a trait has various graphical representations so it can be manipulated
# in a GUI (via the traitsui package).


# The first set of classes define what kind of filtering happens on the visible data. It's often wise to define a
# superclass, even if it does nothing except name the overall purpose of the group of classes. If anything occurs
# later that should be implemented for all filters, it can be done in one place here.
class AppliesSeriesFiltering(HasTraits):
    """These classes do the actual filtering"""

    # Traits are (most commonly) defined using class member syntax (i.e. at the class level).
    # They are typed according to type names from traits.api (usually upper-case versions of Python types).

    # All filters can make use of the sampling rate
    sample_rate = Float
    # let the parent object know whether there are any config parameters to display in GUI
    has_params = Bool(False)

    # default (blank) View
    def default_traits_view(self):
        return View()

    def apply(self, curve_collection):
        y = curve_collection.current_data(visible=False)[1]
        y = self(y)
        # set data while temporarily disabling the collection listening while data changes
        curve_collection.set_external_data(y, visible=False, redraw=True)

# A dummy filter when no filter is selected (should probably never be called)
class NoFilter(AppliesSeriesFiltering):

    def __call__(self, array):
        print('warning -- no-op filter is applied and probably should not be')
        return array


# An actual filter that takes the abs-value of the Hilbert transform to estimate a power envelope
class HilbertEnvelope(AppliesSeriesFiltering):

    # This defines what happens when a HilbertEnvelope object is called like a function
    def __call__(self, array):
        n = array.shape[1]
        nfft = nextpow2(n)
        xa = hilbert(array, N=nfft, axis=-1)
        return np.abs(xa[:, :n])


# Very simple bandpass -> rectify -> smooth transform
class RectifyAndSmooth(AppliesSeriesFiltering):
    has_params = True
    bandpass = Str
    f_lo = Property(Float, depends_on='bandpass')
    f_hi = Property(Float, depends_on='bandpass')
    square = Bool(True)
    tau = Float
    para = Bool(False)
    filtfilt = Bool(False)

    def _parse_bandpass(self):
        lo = -1
        hi = -1
        if len(self.bandpass):
            split = self.bandpass.split(',')
            try:
                lo = float(split[0])
            except:
                pass
            try:
                hi = float(split[1])
            except:
                pass
        return lo, hi

    def _get_f_lo(self):
        return self._parse_bandpass()[0]

    def _get_f_hi(self):
        return self._parse_bandpass()[1]

    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(Label('Band (comma-sep)'), UItem('bandpass')),
                VGroup(Label('Square?'), UItem('square')),
                VGroup(Label('Gauss tau (s)'), Label('(Or tau=0 to skip rectifier)'), UItem('tau')),
                VGroup(Label('Theaded?'), UItem('para'),
                       Label('Zero-phase?'), UItem('filtfilt'))
            )
        )
        return v

    def __call__(self, array):
        if self.f_lo > 0 or self.f_hi > 0:
            block_filter = 'parallel' if self.para else 'serial'
            fdes = dict(lo=self.f_lo, hi=self.f_hi, Fs=self.sample_rate, ord=3)
            farg = dict(filtfilt=self.filtfilt)
            y = filter_array(array, inplace=False, block_filter=block_filter, design_kwargs=fdes, filt_kwargs=farg)
        tau_samps = self.tau * self.sample_rate
        if tau_samps == 0:
            return y
        if self.square:
            y **= 2
        else:
            np.abs(y, out=y)
        if tau_samps > 0:
            if self.para:
                gauss_para = split_at()(gaussian_filter1d)
                y = gauss_para(y, tau_samps, axis=1)
            else:
                y = gaussian_filter1d(y, tau_samps, axis=-1)
        if self.square:
            np.sqrt(y, out=y)
        return y


# This is a slightly more tunable power envelope calculator. Unlike the Hilbert transform, this estimator can make
# narrowband envelopes on broadband series. You can control the bandwidth and center frequency of
# the power estimate before demodulating to baseband. All these parameters are defined as traits on the object.
class MultitaperDemodulate(AppliesSeriesFiltering):
    has_params = True
    BW = Float(10)  # half-bandwidth of projection in Hz
    f0 = Float(0)  # center frequency of bandpass
    bandpass = Tuple(-5, 5)  # Can also set up filter with a bandpass
    N = Int(100)  # segment size (moving projection slides this window over the whole series)
    NW = Property(Float, depends_on='BW, N, moving')  # This property updates the order of the DPSS projectors
    # Moving filters are good for longer windows, but a single filter projection is efficient for short windows
    moving = Bool(True)
    baseband = Bool(True)  # Demodulate bandpass filter (shift to baseband for power envelope)
    Kmax = Int
    ignore = ToggleState(init_state=False)

    # NW is a "property", which is dynamically calculated based on the values of other traits. It gets updated when
    # any of the traits it depends on change.
    def _get_NW(self):
        if self.moving:
            N = self.N
        else:
            N = self.curve_manager.source_curve.y_visible.shape[1]
        return bw2nw(self.BW, self.N, self.sample_rate, halfint=True)

    @observe('BW, f0, bandpass')
    def _sync_filter_spec(self, event):
        if self.ignore:
            return
        if event.name == 'bandpass':
            lo, hi = self.bandpass
            with self.ignore(True):
                self.f0 = (lo + hi) / 2
                self.BW = (hi - lo) / 2
        elif event.name in ('BW', 'f0'):
            lo = self.f0 - self.BW
            hi = self.f0 + self.BW
            with self.ignore(True):
                self.bandpass = lo, hi

    def __call__(self, array):
        nx = array.shape[1]
        # Setting a concentration threshold of 1 - 10^-n only shaves off ~n modes.
        # Let's set an arbitrary minimum of 10 modes (plus 3 for a safety factor, since
        # the concentration-to-mode relationship is not perfect).
        n_play = min(1, max(6, 2 * self.NW - 10 - 3))
        spectral_concentration = 1 - 10 ** (-n_play)
        # if the window is shorter than N, just make a non-moving projection
        if nx <= self.N or not self.moving:
            Kmax = self.Kmax if self.Kmax else None
            baseband = slepian_projection(array, self.BW, self.sample_rate, w0=self.f0, Kmax=Kmax,
                                          min_conc=spectral_concentration,
                                          baseband=self.baseband,
                                          onesided=self.baseband)
        else:
            baseband = moving_projection(array, self.N, self.BW, Fs=self.sample_rate, f0=self.f0,
                                         min_conc=spectral_concentration, baseband=self.baseband)
        # if in Single-side baseband mode, return the magnitude
        if self.baseband:
            baseband = np.abs(baseband)
        return baseband

    # A HasTraits object can expose its traits for GUI manipulation using a View. The MultitaperDemodulate object
    # will have a horizontal panel with entry boxes (default input method for numbers). The View has a single HGroup
    # element, which holds all the traits. Each traits is exposed as an unlabeled box stacked with a descriptive
    # label above it in a VGroup.
    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(
                    HGroup(view_label_item('BW', 'half-BW (Hz)'),
                           view_label_item('f0', 'center freq (Hz)')),
                    view_label_item('bandpass', 'bandpass', vertical=False)
                ),
                VGroup(view_label_item('moving', 'Slide win'),
                       view_label_item('baseband', 'Demodulate'),
                       view_label_item('N', 'Win size (pts)', enabled_when='object.moving')),
                VGroup(view_label_item('Kmax', 'Kmax'),
                       view_label_item('NW', 'DPSS ord.', item_kwargs=dict(style='readonly')))
            )
        )
        return v


# The previous filters work by replacing on-screen signals in a PlotCurveCollection. As an alternative,
# plot both signals on screen using a FollowerCollection.
class FilterFollower(SelectedFollowerCollection):
    data_filtered = QtCore.Signal(QtCore.QObject)

    def __init__(self, curve_collection: PlotCurveCollection, transform: AppliesSeriesFiltering, **kwargs):
        kwargs['clickable'] = False
        if isinstance(curve_collection, SelectedFollowerCollection):
            super().__init__(curve_collection._source, **kwargs)
        else:
            super().__init__(curve_collection, **kwargs)
        self.signal_transform = transform
        self._y_filtered = None
        self.register_connection(self._source.data_changed, self.transform_page)
        if isinstance(curve_collection, SelectedFollowerCollection):
            self._active_channels = curve_collection._active_channels.copy()
            self.register_connection(curve_collection.selection_changed, self.transform_page)
        self.data_filtered.connect(self._source_triggered_update)
        # This should be ready at the start
        self.transform_page(curve_collection)

    def data_ready(self):
        # Check that the source data_changed signal has triggered a signal transform
        paged = self._cslice == self._source._cslice
        if not paged:
            return False
        # Check if channels have been activated/deactivated but they're not visible yet (1)
        n_active = self._active_channels.sum()
        if n_active > 0 and self.y_visible is None:
            return False
        # Check if channels have been activated/deactivated but they're not visible yet (2)
        if n_active > 0 and len(self.y_visible) != n_active:
            return False
        return True

    def transform_page(self, source: Union[PlotCurveCollection, np.ndarray]):
        if isinstance(source, np.ndarray):
            self._active_channels = source.copy()
        if not self._active_channels.any():
            self._y_filtered = None
        else:
            raw_signal = self._source.y_slice[self._active_channels]
            self._y_filtered = self.signal_transform(raw_signal)
        # following this attribute will signal that the filter data is current
        self._cslice = self._source._cslice
        self.data_filtered.emit(self)

    @property
    def y_visible(self):
        if not self._active_channels.any():
            return None
        # This case might catch a transient state going between 0 to >0 active channels
        if self._y_filtered is None:
            return None
        full_start = self._source._cslice.start
        full_stop = self._source._cslice.stop
        page_start, page_stop = self._source._view_range()
        # This should probably not happen, since the follower is only activated after
        # the source data is ready. But best to check.
        if page_start < full_start or page_stop > full_stop:
            return None
        x1 = page_start - full_start
        x2 = page_stop - full_start
        # _y_filtered is already a subset of source channels
        return self._y_filtered[:, x1:x2]

    @y_visible.setter
    def y_visible(self, new):
        return

    def updatePlotData(self, data_ready=False):
        # This update is ALSO gated on whether the filtered data is ready,
        # since the source-following callbacks might be out of order.
        if not (self.can_update and self.data_ready()):
            return
        super().updatePlotData(data_ready=True)


# A "Handler" in Traits is an object that responds to user input when the response behavior is somewhat non-trivial.
# In this case, the filtering module will have a choice of filters that can be applied. Based on the GUI manipulation
# of that choice, a different filter calculator will be constructed (the AppliesSeriesFiltering types defined above)
# and attached to the abstract "filter" definition
class SeriesFilterHandler(Handler):

    # This method will be triggered when the "name" trait is changed on the Handler's "object". The call signature
    # takes a UI info object "info" as an argument.
    def object_name_changed(self, info):
        name = info.object.name.lower()
        if name == 'hilbert envelope':
            info.object.filter = HilbertEnvelope(sample_rate=info.object.sample_rate)
        elif name == 'mult.taper demodulate':
            info.object.filter = MultitaperDemodulate(sample_rate=info.object.sample_rate)
        elif name == 'rectify & smooth':
            info.object.filter = RectifyAndSmooth(sample_rate=info.object.sample_rate)
        elif name == 'none':
            info.object.filter = NoFilter()


# This is the GUI abstraction of the filter on the visible series. It has a name that is set by interacting with the
# module (below). The "handler" above attaches a concrete filter calculator based on the name.
class SeriesFilter(HasTraits):
    name = Str
    filter = Instance(AppliesSeriesFiltering)
    sample_rate = Float

    view = View(
        HGroup(
            UItem('name', style='readonly'),
            # The filter types will be displayed in "custom" mode (each filter presents a different menu).
            # It would be nice to only display the filter config panel only when there are parameters,
            # but it looks like the Handler never triggers if this part is invisible
            Group(UItem('filter', style='custom')),  # , visible_when='filter.filter.has_params')
        ),
        # Specify that this view is being monitored by a handler
        handler=SeriesFilterHandler
    )


# Name the available filters
series_filters = ('None', 'Hilbert envelope', 'Rectify & smooth', 'Mult.Taper demodulate')


# This is the actual module class that is represented as a panel in the main GUI window. It inherits from VisModule,
# which defines some common information for modules (e.g. the "parent", which is the object controlling the entire main
# window).
class SeriesFiltering(VisModule):
    # all modules have a name (appears as the tab name)
    name = 'Series Filters'
    # this is a general abstract filter that can be attached to the visible data
    filter = Instance(SeriesFilter)
    # The filter type is an Enum. An Enum presents itself graphically as a drop-down list of options. But the Trait
    # is only one value at a time. Here it is initialized as 'None' and can take on values from the set values in
    # series_filters.
    filter_type = Enum(series_filters[0], series_filters)
    # The name of the applied filter
    applied_filter = Str('None')
    # Option to replace on-page data (default True)
    replace_data = Bool(True)
    # Option to filter selected signals (default False)
    selected_filter = Bool(False)
    # A Button is a non-Python type with pretty obvious behavior (it can be "fired" when pressed)
    set_filter = Button('Set window filter')
    unset_filter = Button('Remove window filter')
    # not really a trait, just housekeeping
    _cb_connection = None

    # One way to respond to a change of value is to define the method "_<trait-name>_changed(self)"
    def _filter_type_changed(self):
        # Every time the filter type changes from the drop-down menu,
        # change the current filter configuration object. This *does not*
        # change the applied filter -- need to hit "set filter" button to do that.
        rate = self.parent.x_scale ** -1.0
        self.filter = SeriesFilter(name=self.filter_type, sample_rate=rate)

    # Button response behavior defined with "_<button-name>_fired(self)" method.
    def _set_filter_fired(self):
        self.disconnect_filter(trigger_redraw=False)
        if self.filter is None or self.filter.name == 'None':
            # this is the same as unsetting a filter, which has already been done
            return
        # Decide which filter type to apply
        if self.replace_data:
            self.set_inplace_filter()
        else:
            self.add_filter_curves()

    def set_inplace_filter(self):
        # get the concrete filter calculator from the current abstract filter
        filter = self.filter.filter
        # attach a new debounce filter to this object (if a reference it not saved, it will get garbage-collected!)
        # We're going to attach this debounced callback to the curve collection that watches the zoom-plot data. The
        # curve collection is a part of the PyQtGraph panel system that underlies the main plot elements.
        curve_collection = self.curve_manager.source_curve
        # connect callback to the signal
        self._cb_connection = DebounceCallback.connect(curve_collection.data_changed, filter.apply)
        # apply the filter and cause the signal to emit
        filter.apply(curve_collection)
        # redraw the raw data and emit the data changed signal
        # curve_collection.redraw_data(ignore_signals=False)
        # Now that a filter is applied, set the name for display
        self.applied_filter = self.filter.name

    def add_filter_curves(self):
        transform = self.filter.filter
        clickable = self.selected_filter
        if clickable:
            source = self.curve_manager.curves_by_name('selected')
        else:
            source = self.curve_manager.source_curve
        pen_args = dict(color='r', width=1)
        filter_follower = FilterFollower(source, transform,
                                         init_active=not clickable,
                                         pen_args=pen_args,
                                         shadowpen_args=dict())
        self.curve_manager.add_new_curves(filter_follower, 'filter tab')
        filter_follower.data_filtered.emit(filter_follower)

    def disconnect_filter(self, trigger_redraw=True):
        # If a connection is present, disconnect it and redraw the raw data (without signaling data change)
        if self._cb_connection is not None:
            self.curve_manager.source_curve.data_changed.disconnect(self._cb_connection)
            self.curve_manager.source_curve.unset_external_data(visible=False, redraw=trigger_redraw)
            self._cb_connection = None
        if 'filter tab' in self.curve_manager._curve_names:
            self.curve_manager.remove_curves('filter tab')

    def _unset_filter_fired(self):
        # unset the filter name and callback
        self.disconnect_filter(trigger_redraw=True)
        self.applied_filter = 'None'

    # The view on this module is deceptively simple. In a horizontal span (HGroup), show
    # 1) a group with the filter type options and the set/unset buttons
    # 2) the abstract filter itself in custom mode, since the GUI panel is adaptive. (only show if filter is not None)
    # 3) the currently applied filter (possibly "None")
    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(UItem('filter_type'),
                       HGroup(Label('Replace data'), UItem('replace_data')),
                       HGroup(Label('Filter selected'), UItem('selected_filter'), enabled_when='not replace_data'),
                       UItem('set_filter'),
                       UItem('unset_filter')),
                Group(UItem('filter'), style='custom', label='Configure filter',
                      visible_when='filter and filter.name != "None"'),
                Group(Label('Applied filter:'), UItem('applied_filter')),
                springy=True
            ),
            title=self.name
        )
        return v





