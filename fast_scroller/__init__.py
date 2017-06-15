"""
New timeseries scrolling tool with baseline analysis modules.
"""

import os
# these imports set up some qt runtime stuff (??)
import PySide
import pyqtgraph as pg
pg.QtGui.QApplication.setGraphicsSystem("raster")

from traits.api import HasTraits, Str
from traitsui.api import View, VGroup, spring, HGroup, Item

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
