import h5py
from traits.api import Button, Str, File
from traitsui.api import View, Group, UItem, Label, HGroup, EnumEditor, VGroup

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from ecoglib.trigger_fun import process_trigger

from .base import VisModule

__all__ = ['Timestamps']

class Timestamps(VisModule):
    name = Str('Time Stamps')

    file = File
    timing_array = Str
    timing_channels = Str('0')

    load = Button('Show timestamps')
    unload = Button('Hide timestamps')

    def _get_timestamps(self):
        array = h5py.File(self.file, 'r')[self.timing_array]
        chans = map(int, self.timing_channels.split(','))
        time_stamps, _ = process_trigger( array[chans] )
        time_stamps = time_stamps.astype('d') * self.parent.x_scale
        return time_stamps

    def _load_fired(self):
        if not hasattr(self, 'stamps'):
            self.stamps = dict()

        if self.file not in self.stamps:
            # load new timestamps and hide old timestamps
            self._unload_fired()
            self.stamps[self.file] = self._get_timestamps()
            N = len(self.stamps[self.file])
            self.p1_lines = [None for x in xrange(N)]
            self.p2_lines = [None for x in xrange(N)]

        # connect a callback that adds only the needed lines
        # when the scroller's xrange changes
        p1 = self.parent._qtwindow.p1
        p1.sigRangeChanged.connect(self._draw_lines_in_p1_range)
        p1.sigRangeChanged.emit(None, p1.viewRange())
        p2 = self.parent._qtwindow.p2
        p2.sigRangeChanged.connect(self._draw_lines_in_p2_range)
        p2.sigRangeChanged.emit(None, p2.viewRange())

    def _draw_lines_in_p1_range(self, window, vrange):
        plot = self.parent._qtwindow.p1
        self._draw_lines_in_range(plot, self.p1_lines, vrange[0])

    def _draw_lines_in_p2_range(self, window, vrange):
        plot = self.parent._qtwindow.p2
        self._draw_lines_in_range(plot, self.p2_lines, vrange[0])

    def _draw_lines_in_range(self, plot, line_set, x_range):
        ts = self.stamps[self.file]
        i1, i2 = ts.searchsorted(x_range)
        pen = pg.mkPen('r', width=1, style=QtCore.Qt.DashLine)
        for n in xrange(max(0, i1-1), i2):
            if line_set[n] is not None:
                line_set[n].show()
                continue
            mark = pg.InfiniteLine(angle=90, pos=ts[n], pen=pen)
            plot.addItem(mark)
            line_set[n] = mark

    def _unload_fired(self):
        if hasattr(self, 'p1_lines'):
            for line in self.p1_lines:
                if line is not None:
                    line.hide()
            p1 = self.parent._qtwindow.p1
            try:
                p1.sigRangeChanged.disconnect(self._draw_lines_in_p1_range)
            except RuntimeError:
                pass
            vb = p1.getViewBox()
            vb.sigStateChanged.emit(vb)
        if hasattr(self, 'p2_lines'):
            for line in self.p2_lines:
                if line is not None:
                    line.hide()
            p2 = self.parent._qtwindow.p2
            try:
                p2.sigRangeChanged.disconnect(self._draw_lines_in_p2_range)
            except RuntimeError:
                pass
            vb = p2.getViewBox()
            vb.sigStateChanged.emit(vb)

    def default_traits_view(self):
        from ..data_files import FileHandler
        v = View(
            HGroup(
                Group(Label('HDF5 file with timing signal'),
                      UItem('file')),
                Group(Label('Array with timing channel(s)'),
                      UItem('timing_array',
                            editor=EnumEditor(name='handler.fields'))),
                Group(Label('Channel(s) with timing signal (comma-sep)'),
                      UItem('timing_channels')),
                VGroup(
                    UItem('load', enabled_when='len(timing_source) > 0'),
                    UItem('unload', enabled_when='len(timing_source) > 0'))
                ),
            handler=FileHandler
            )
        return v
