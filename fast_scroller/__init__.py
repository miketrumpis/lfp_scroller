"""
New timeseries scrolling tool with baseline analysis modules.
"""

from distutils.version import StrictVersion
import matplotlib
import pyqtgraph as pg


__version__ = '0.1.0'
qt_version = pg.Qt.VERSION_INFO.split()[-1]
QT = ''
if StrictVersion('4.5.0') <= StrictVersion(qt_version) < StrictVersion('5.0.0'):
    QT = 'qt4'
    pg.Qt.QtWidgets.QApplication.setGraphicsSystem('raster')
    matplotlib.use('Qt4Agg')
elif StrictVersion('5.0.0') <= StrictVersion(qt_version) < StrictVersion('6.0.0'):
    QT = 'qt5'
    matplotlib.use('Qt5Agg')
else:
    QT = 'qt6'
    matplotlib.use('Qt5Agg')

    def exec_(obj):
        return obj.exec()

    for o in dir(pg.Qt.QtWidgets):
        obj = getattr(pg.Qt.QtWidgets, o)
        if hasattr(obj, 'exec'):
            obj.exec_ = exec_
            print(f'patched {o}')
    # pg.Qt.QtWidgets.QApplication.exec_ = exec_
    # pg.Qt.QtWidgets.QFileDialog.exec_ = exec_
    # pg.Qt.QtWidgets.QDialog.exec_ = exec_

    # import pyface.util.guisupport as gs
    # def start_event_loop_qt4(app=None):
    #     """Start the qt4 event loop in a consistent manner."""
    #     if app is None:
    #         app = gs.get_app_qt4([""])
    #     if not gs.is_event_loop_running_qt4(app):
    #         app._in_event_loop = True
    #         app.exec()
    #         app._in_event_loop = False
    #     else:
    #         app._in_event_loop = True
    # print('applying monkey patch')
    # gs.start_event_loop_qt4 = start_event_loop_qt4
    #
    # from pyface.ui.qt4.dialog import Dialog, _RESULT_MAP
    # from pyface.qt import QtCore, QtGui
    # def _show_modal(obj):
    #     dialog = obj.control
    #     dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    #
    #     # Suppress the context-help button hint, which
    #     # results in a non-functional "?" button on Windows.
    #     dialog.setWindowFlags(
    #         dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
    #     )
    #
    #     retval = dialog.exec()
    #     return _RESULT_MAP[retval]
    # Dialog._show_modal = _show_modal
    # print('monkey patching show_modal')

print('qt version is', qt_version, 'Qt is', QT)


# This was some nonsense to force Pyside2 to load if the Windows OS was quite old
#
# these imports set up some qt runtime stuff
# pyqtgraph 0.12.x does now supports PySide2, and there is a bad mixture
# with windows 8, PyQt5, and traitsui -- so prefer to load PySide2 first
# old_windows = platform.system() == 'Windows' and StrictVersion(platform.version()) < StrictVersion('10.0.0')
# Also respect the QT_API environment variable to load PySide2
# pyside2_env = 'QT_API' in os.environ and os.environ['QT_API'].lower() == 'pyside2'
# pyside2_set = False
# if pyside2_env or old_windows:
#     try:
#         import PySide2.QtCore
#         pyqtSlot = PySide2.QtCore.Slot
#         pyqtSignal = PySide2.QtCore.Signal
#         pyside2_set = True
#     except ImportError:
#         pyside2_set = False
# if not pyside2_set:
#     import PyQt5.QtCore
#     pyqtSlot = PyQt5.QtCore.pyqtSlot
#     pyqtSignal = PyQt5.QtCore.pyqtSignal

