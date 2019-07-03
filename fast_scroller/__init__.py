"""
New timeseries scrolling tool with baseline analysis modules.
"""

from distutils.version import StrictVersion
# these imports set up some qt runtime stuff
# prefer PySide (?) but fall back on default PyQtx
try:
    import PySide
except ImportError:
    pass
import matplotlib
import pyqtgraph as pg
qt_version = pg.Qt.VERSION_INFO.split()[-1]
if StrictVersion('4.5.0') <= StrictVersion(qt_version) < StrictVersion('5.0.0'):
    pg.QtGui.QApplication.setGraphicsSystem('raster')
    matplotlib.use('Qt4Agg')
else:
    matplotlib.use('Qt5Agg')
import pyface
pyf_version = StrictVersion(pyface.__version__)
pyf_new_api = pyf_version >= StrictVersion('6.1.0')
