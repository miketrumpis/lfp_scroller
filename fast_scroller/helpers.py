"""Module for helpful code (mostly traits-based mods)."""

import os
from traits.api import HasTraits, Str
from traitsui.api import View, VGroup, spring, HGroup, Item


# A HasTraits "mixin" that keeps a reference to its own UI window so that
# QT doesn't destroy the window if it goes out of focus.
# Note: implementing as a direct subclass (rather than mixin) to avoid
# weird multiple inheritance metaclass conflicts.
# To do correctly: http://mcjeff.blogspot.com/2009/05/odd-python-errors.html
class PersistentWindow(HasTraits):

    def edit_traits(self, **args):
        ui = super(PersistentWindow, self).edit_traits(**args)
        self._hold_ui = ui
        def del_ui():
            delattr(self, '_hold_ui')
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


def validate_file_path(f):
    f = os.path.abspath(f)
    exists = os.path.exists(os.path.dirname(f))
    not_dir = not os.path.isdir(f)
    return exists and not_dir