"""
New timeseries scrolling tool with baseline analysis modules.
"""

import platform
import multiprocessing
# fork is nicer than spawn with GUIs
if platform.system() != 'Windows':
    multiprocessing.set_start_method('fork')
from distutils.version import StrictVersion
# these imports set up some qt runtime stuff
# pyqtgraph does not seem to support PySide2, so prefer PyQt5
try:
    import PyQt5
except ImportError:
    pass
import matplotlib
import pyqtgraph as pg
qt_version = pg.Qt.VERSION_INFO.split()[-1]
if StrictVersion('4.5.0') <= StrictVersion(qt_version) < StrictVersion('5.0.0'):
    QT5 = False
    pg.QtGui.QApplication.setGraphicsSystem('raster')
    matplotlib.use('Qt4Agg')
else:
    QT5 = True
    matplotlib.use('Qt5Agg')
import pyface
pyf_version = StrictVersion(pyface.__version__)
pyf_new_api = pyf_version >= StrictVersion('6.1.0')
