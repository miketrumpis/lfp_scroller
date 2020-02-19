import h5py
from traits.api import Button, Str, File, Bool, HasTraits, List, Instance, on_trait_change
from traitsui.api import View, Group, UItem, Label, HGroup, EnumEditor, VGroup, TableEditor
from traitsui.table_column import ObjectColumn
from traitsui.extras.checkbox_column import CheckboxColumn

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from ecogdata.trigger_fun import process_trigger

from .base import VisModule

__all__ = ['Timestamps']

color_cycle = {'r', 'c', 'm', 'y', 'g', 'b'}


class TimeStampSet(HasTraits):
    parent = Instance('fast_scroller.new_scroller.VisWrapper')
    file = File(exists=True)
    name = Str
    color = Str('r')
    timing_array = Str
    timing_channels = Str('0')
    is_binary = Bool(True)
    show = Bool(False)

    #load = Button('Show timestamps')
    #unload = Button('Hide timestamps')

    def _get_timestamps(self):
        array = h5py.File(self.file, 'r')[self.timing_array]
        if array.shape[-1] == array.size:
            chans = array[:].squeeze()
        else:
            chans = list(map(int, self.timing_channels.split(',')))
            chans = array[chans]
        if self.is_binary:
            time_stamps, _ = process_trigger(chans)
            time_stamps = time_stamps.astype('d') * self.parent.x_scale
        else:
            time_stamps = chans.astype('d') * self.parent.x_scale
        return time_stamps

    @on_trait_change('show')
    def _toggle_visible(self):
        if self.show:
            self._load()
        else:
            self._unload()

    def _load(self):
        if not hasattr(self, 'stamps'):
            self.stamps = dict()

        if self.file not in self.stamps:
            # load new timestamps and hide old timestamps
            self._unload()
            self.stamps[self.file] = self._get_timestamps()
            N = len(self.stamps[self.file])
            self.p1_lines = [None for x in range(N)]
            self.p2_lines = [None for x in range(N)]

        # connect a callback that adds only the needed lines
        # when the scroller's range changes
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
        #pen = pg.mkPen('r', width=1, style=QtCore.Qt.DashLine)
        pen = pg.mkPen(self.color, width=1, style=QtCore.Qt.DashLine)
        for n in range(max(0, i1-1), i2):
            if line_set[n] is not None:
                line_set[n].show()
                continue
            mark = pg.InfiniteLine(angle=90, pos=ts[n], pen=pen)
            plot.addItem(mark)
            line_set[n] = mark

    def _unload(self):
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


tab_editor = TableEditor(
    columns=[
        ObjectColumn(name='name', label='Name', editable=True),
        ObjectColumn(name='color', label='Color', editable=True),
        CheckboxColumn(name='is_binary', label='Binary?',
                       editable=True, width=0.1),
        CheckboxColumn(name='show', label='Visible', editable=True, width=0.1)
    ],
    auto_size=False,
    row_factory=TimeStampSet,
    selected='selected_set'
)


class Timestamps(VisModule):
    name = Str('Time Stamps')
    file = File(exists=True)
    timing_array = Str
    timing_channels = Str('0')
    time_set = List(TimeStampSet)
    selected_set = Instance(TimeStampSet)
    add_set = Button('Load timestamps')
    pop_set = Button('Remove timestamps')

    def _add_set_fired(self):
        used_colors = set([s.color for s in self.time_set])
        unused_colors = color_cycle - used_colors
        clr = list(unused_colors)[0]
        new_set = TimeStampSet(
            name='Set ' + str(len(self.time_set) + 1),
            color=clr, file=self.file, timing_array=self.timing_array,
            timing_channels=self.timing_channels, parent=self.parent
        )
        self.time_set.append(new_set)

    def _pop_set_fired(self):
        set = self.selected_set
        set._unload()
        self.selected_set = None
        self.time_set.remove(set)

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
                ## Item('is_binary', label='Binary series?'),
                VGroup(
                    UItem('add_set', enabled_when='len(timing_source) > 0'),
                    UItem('pop_set', enabled_when='len(timing_source) > 0')
                ),
                Group(
                    UItem('time_set', editor=tab_editor),
                    show_border=True
                ),
            ),

            handler=FileHandler
        )
        return v
