from time import time
import numpy as np
from scipy.signal import hilbert
from traits.api import HasTraits, Instance, Button, Int, Enum, Float, Str, Bool, Property
from traitsui.api import View, UItem, Handler, Group, HGroup, VGroup, Label
from pyface.timer.api import do_after

from ecoglib.estimation.multitaper import bw2nw
from ecogdata.util import nextpow2
from ecogdata.parallel.array_split import timestamp
from ecogdata.parallel.mproc import get_logger
from ecogdata.filt.time import moving_projection, slepian_projection

from .base import VisModule
from ..helpers import DebounceCallback


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

    # let the parent object know whether there are any config parameters to display in GUI
    has_params = Bool(False)

    # default (blank) View
    def default_traits_view(self):
        return View()

    def apply(self, curve_collection):
        y = curve_collection.current_data(visible=False)[1]
        y = self(y)
        # set data while temporarily disabling the collection listening while data changes
        curve_collection.set_data_on_collection(y, ignore_signals=True, visible=False)

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


# This is a slightly more tunable power envelope calculator. Unlike the Hilbert transform, this estimator can make
# narrowband envelopes on broadband series. You can control the bandwidth and center frequency of
# the power estimate before demodulating to baseband. All these parameters are defined as traits on the object.
class MultitaperDemodulate(AppliesSeriesFiltering):
    has_params = True
    BW = Float(0.05)  # half-bandwidth of projection in Hz
    Fs = Float(1)  # full bandwidth of series
    f0 = Float   # center frequency of bandpass
    N = Int(100)  # segment size (moving projection slides this window over the whole series)
    NW = Property(Float, depends_on='BW, N, Fs')  # This property updates the order of the DPSS projectors

    # NW is a "property", which is dynamically calculated based on the values of other traits. It gets updated when
    # any of the traits it depends on change.
    def _get_NW(self):
        return bw2nw(self.BW, self.N, self.Fs, halfint=True)

    def __call__(self, array):
        nx = array.shape[1]
        # if the window is shorter than N, just make a non-moving projection
        if nx <= self.N:
            baseband = slepian_projection(array, self.BW, self.Fs, w0=self.f0, baseband=True, onesided=True)
        else:
            baseband = moving_projection(array, self.N, self.BW, Fs=self.Fs, f0=self.f0, baseband=True)
        return np.abs(baseband)

    # A HasTraits object can expose its traits for GUI manipulation using a View. The MultitaperDemodulate object
    # will have a horizontal panel with entry boxes (default input method for numbers). The View has a single HGroup
    # element, which holds all the traits. Each traits is exposed as an unlabeled box stacked with a descriptive
    # label above it in a VGroup.
    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(Label('half-BW (Hz)'), UItem('BW')),
                VGroup(Label('samp freq (Hz)'), UItem('Fs')),
                VGroup(Label('center freq (Hz)'), UItem('f0')),
                VGroup(Label('Win size (pts)'), UItem('N')),
                VGroup(Label('DPSS ord.'), UItem('NW', style='readonly'))
            )
        )
        return v


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
            info.object.filter = HilbertEnvelope()
        elif name == 'mult.taper demodulate':
            info.object.filter = MultitaperDemodulate()
        elif name == 'none':
            info.object.filter = NoFilter()


# This is the GUI abstraction of the filter on the visible series. It has a name that is set by interacting with the
# module (below). The "handler" above attaches a concrete filter calculator based on the name.
class SeriesFilter(HasTraits):
    name = Str
    filter = Instance(AppliesSeriesFiltering)

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
series_filters = ('None', 'Hilbert envelope', 'Mult.Taper demodulate')


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
        self.filter = SeriesFilter(name=self.filter_type)

    # Button response behavior defined with "_<button-name>_fired(self)" method.
    def _set_filter_fired(self):
        if self.filter is None or self.filter.name == 'None':
            # this is the same as unsetting a filter
            self.filter = None
            self._unset_filter_fired()
            return
        if self._cb_connection is not None:
            self.curve_collection.data_changed.disconnect(self._cb_connection)

        # get the concrete filter calculator from the current abstract filter
        filter = self.filter.filter
        # attach a new debounce filter to this object (if a reference it not saved, it will get garbage-collected!)
        # We're going to attach this debounced callback to the curve collection that watches the zoom-plot data. The
        # curve collection is a part of the PyQtGraph panel system that underlies the main plot elements.
        curve_collection = self.curve_collection
        # connect callback to the signal
        self._cb_connection = DebounceCallback.connect(curve_collection.data_changed, filter.apply)
        # apply the filter and cause the signal to emit
        filter.apply(curve_collection)
        # redraw the raw data and emit the data changed signal
        # curve_collection.redraw_data(ignore_signals=False)
        # Now that a filter is applied, set the name for display
        self.applied_filter = self.filter.name

    def _unset_filter_fired(self):
        # If a connection is present, disconnect it and redraw the raw data (without signaling data change)
        if self._cb_connection is not None:
            self.curve_collection.data_changed.disconnect(self._cb_connection)
            self._cb_connection = None
            self.curve_collection.unset_data_on_collection(ignore_signals=True, visible=False)
        # unset the filter name and callback
        self.applied_filter = 'None'

    # The view on this module is deceptively simple. In a horizontal span (HGroup), show
    # 1) a group with the filter type options and the set/unset buttons
    # 2) the abstract filter itself in custom mode, since the GUI panel is adaptive. (only show if filter is not None)
    # 3) the currently applied filter (possibly "None")
    def default_traits_view(self):
        v = View(
            HGroup(
                Group(UItem('filter_type'), UItem('set_filter'), UItem('unset_filter')),
                Group(UItem('filter'), style='custom', label='Configure filter',
                      visible_when='filter and filter.name != "None"'),
                Group(Label('Applied filter:'), UItem('applied_filter')),
                springy=True
            ),
            title=self.name
        )
        return v





