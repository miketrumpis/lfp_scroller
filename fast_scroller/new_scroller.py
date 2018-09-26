from __future__ import division

from traits.api import HasTraits, Instance, Float, Enum, \
     Any, List, on_trait_change, Event
from traitsui.api import View, UItem, VSplit, CustomEditor, \
     HSplit, Group, Label, ListEditor

import matplotlib as mpl
mpl.rcParams['interactive'] = True
mpl.use('Qt4Agg')
from pyqtgraph.Qt import QtGui

from .modules import *

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    y_spacing = Enum(0.2, [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    chan_map = Any
    region = Event
    modules = List( [IntervalTraces,
                     AnimateInterval,
                     IntervalSpectrum,
                     IntervalSpectrogram,
                     SpatialVariance] )

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)
        # connect the region-changed signal to signal a traits callback
        qtwindow.region.sigRegionChanged.connect(self._region_did_change)
        self._make_modules()

    def _region_did_change(self):
        self.region = True

    def _make_modules(self):
        mods = self.modules[:]
        self.modules = []
        for m in mods:
            self.modules.append( m(parent=self) )

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing * 1e-3)

    def default_traits_view(self):
        ht = 1000
        v = View(
            VSplit(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow.win),
                      resizable=True,
                      #height=(ht-150)),
                      height=0.85),
                HSplit(
                    Group(Label('Trace spacing (mv)'),
                          UItem('y_spacing', resizable=True)),
                                
                    UItem('modules', style='custom', resizable=True,
                          editor=ListEditor(use_notebook=True,
                                            deletable=False, 
                                            dock_style='tab',
                                            page_name='.name'),
                                            
                          height=-150),
                      )
                ),
            width=1200,
            height=ht,
            resizable=True,
            title='Quick Scanner'
        )
        return v
    
