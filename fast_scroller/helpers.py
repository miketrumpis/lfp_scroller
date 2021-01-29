"""Module for helpful code (mostly traits-based mods)."""

import os
from time import time
from traits.api import HasTraits, Str
from traitsui.api import View, VGroup, spring, HGroup, Item
from pyface.timer.api import do_after
from ecogdata.parallel.array_split import timestamp
from ecogdata.parallel.mproc import get_logger


# A HasTraits "mixin" that keeps a reference to its own UI window so that
# QT doesn't destroy the window if it goes out of focus.
# Note: implementing as a direct subclass (rather than mixin) to avoid
# weird multiple inheritance metaclass conflicts.
# To do correctly: http://mcjeff.blogspot.com/2009/05/odd-python-errors.html

class _window_holder:

    def __init__(self, window):
        self._hold_ui = window

    def __call__(self):
        delattr(self, '_hold_ui')


class PersistentWindow(HasTraits):

    def edit_traits(self, **args):
        ui = super(PersistentWindow, self).edit_traits(**args)
        del_ui = _window_holder(ui)
        ui.control.destroyed.connect(del_ui)
        return ui


# generic error message pop-up
class Error(HasTraits):
    error_msg = Str('')
    view = View(
        VGroup(
            spring,
            HGroup(Item('error_msg', style='readonly'), show_labels=False),
            spring
            ),
        buttons=['OK']
    )


class DebounceCallback:
    debounce_interval_ms = 200

    def __init__(self, callback):
        """
        A de-bounced dispatcher.

        Parameters
        ----------
        callback: Python callable
            The callable normally attached to signal.connect

        """
        self.callback = callback
        self._signal_times = list()

    @classmethod
    def connect(cls, signal, callback, interval=None):
        debounced = cls(callback)
        if interval is not None:
            debounced.debounce_interval_ms = interval
        signal.connect(debounced.debounced_callback)
        # return the callable that can be disconnected
        return debounced.debounced_callback

    def debounced_callback(self, *emitting):
        """
        This callback can be connected to a signal emitting object.
        When this signal debouncer is idle, the first signal will be dispatched and then follow-up signals
        will be ignored if they follow too closely. A final dispatch will be attempted at the end of the debounce
        interval.

        Parameters
        ----------
        emitting: QtObject
            The emitting object.

        """
        now = time()
        info = get_logger().info
        info('callback for {}, now is {}'.format(self.callback, self._signal_times))
        if not len(self._signal_times):
            # dispatch now and start a new timer
            self.apply_callback(*emitting)
            self._signal_times.append(now)
            do_after(self.debounce_interval_ms, self.apply_callback, *emitting)
        else:
            if (now - self._signal_times[-1]) < self.debounce_interval_ms / 1000:
                # put this time on the end of the stack
                self._signal_times.append(now)
            else:
                # if more than the debounce interval has elapsed, clear the stack
                # and call this callback again
                self._signal_times = list()
                self.debounced_callback(*emitting)

    def apply_callback(self, *emitting):
        """
        Apply the real callback

        Parameters
        ----------
        emitting: QtObject
            The emitting object

        """
        # if the signal time stack is only one deep, then that signal was already dispatched
        if len(self._signal_times) == 1:
            return
        info = get_logger().info
        info('applying callback {} {} {}'.format(self.callback, emitting, timestamp()))
        self.callback(*emitting)
        self._signal_times = list()


def validate_file_path(f):
    f = os.path.abspath(f)
    exists = os.path.exists(os.path.dirname(f))
    not_dir = not os.path.isdir(f)
    return exists and not_dir