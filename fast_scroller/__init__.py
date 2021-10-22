"""
New timeseries scrolling tool with baseline analysis modules.
"""

from distutils.version import StrictVersion
# these imports set up some qt runtime stuff
# pyqtgraph 0.12.x does now supports PySide2, and there is a bad mixture
# with windows 8, PyQt5, and traitsui -- so prefer to load PySide2 first

try:
    import PySide2.QtCore
    pyqtSlot = PySide2.QtCore.Slot
    pyqtSignal = PySide2.QtCore.Signal
except ImportError:
    import PyQt5.QtCore
    pyqtSlot = PyQt5.QtCore.pyqtSlot
    pyqtSignal = PyQt5.QtCore.pyqtSignal

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
