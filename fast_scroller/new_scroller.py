from __future__ import division

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as pp
if __name__=='__main__':
    from matplotlib.figure import Figure
    pyplot_figure = pp.figure
    pp.figure = Figure

import sys
import numpy as np
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Instance, Button, HasTraits, Float, Bool, \
     Enum, on_trait_change, Str, List
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Group, ListEditor

from pyface.timer.api import Timer
import time

from ecogana.anacode.anatool.gui_tools import SavesFigure
from ecogana.anacode.plot_util import filled_interval
from ecoglib.estimation.jackknife import Jackknife
from ecoglib.numutil import nextpow2
from sandbox.split_methods import multi_taper_psd

import seaborn as sns

class PlotsInterval(HasTraits):
    name = Str('__dummy___')
    parent = Instance('VisWrapper')

class IntervalSpectrum(PlotsInterval):
    name = 'Power Spectrum'
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    pow2 = Bool(True)
    avg_spec = Bool(True)
    sem = Bool(True)
    adaptive = Bool(True)
    new_figure = Bool(False)
    plot = Button('Plot')

    def __default_mtm_kwargs(self, n):
        kw = dict()
        kw['NW'] = self.NW
        kw['adaptive'] = self.adaptive
        kw['Fs'] = self.parent.x_scale ** -1.0
        kw['NFFT'] = nextpow2(n) if self.pow2 else n
        kw['jackknife'] = False
        return kw
    
    def spectrum(self, array, **mtm_kw):
        kw = self.__default_mtm_kwargs(array.shape[-1])
        kw.update(mtm_kw)
        fx, pxx, _ = multi_taper_psd(array, **kw)
        return fx, pxx

    def _get_fig(self):
        fig = getattr(self, 'fig', None)
        if fig and not self.new_figure:
            return fig, fig.axes[0], pp.draw
        fig, ax = pp.subplots()
        self._plot_count = 0
        self._colors = sns.color_palette()
        return fig, ax, pp.show
         
    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        fx, pxx = self.spectrum(y)

        fig, ax, update = self._get_fig()
                    
        label = 't ~ {0:.1f}'.format(x.mean())
        
        if self.avg_spec:
            jn = Jackknife(np.log(pxx), axis=0)
            mn, se = jn.estimate(np.mean, se=True)
            if not self.sem:
                # get jackknife stdev
                se = jn.estimate(np.std)
            print mn.shape, se.shape
            filled_interval(
                ax.semilogy, fx, np.exp(mn), np.exp(se),
                color=self._colors[self._plot_count], label=label, ax=ax
                )
            self._plot_count += 1
            ax.legend()
        else:
            lns = ax.semilogy(
                fx, pxx.T, lw=.25, color=self._colors[self._plot_count]
                )
            self._plot_count += 1
            ax.legend(lns[:1], (label,))
        update()

    def default_traits_view(self):
        v = View(
            VGroup(
                HGroup(
                    Item('NW', label='NW'),
                    Item('pow2', label='Use radix-2 FFT'),
                    Item('adaptive', label='Adaptive MTM'),
                    ),
                HGroup(
                    Item('avg_spec', label='Plot avg'),
                    Item('sem', label='Use S.E.M.'),
                    Item('new_figure', label='Plot in new figure')
                    ),
                UItem('plot'),
                label='Spectrum plotting'
                )
            )
        return v

def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 50, 100])
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])
    x_scale = Float

    modules = List( [IntervalSpectrum] )

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)
        self._make_modules()

    def _make_modules(self):
        mods = self.modules[:]
        self.modules = []
        for m in mods:
            self.modules.append( m(parent=self) )

    def __step_frame(self):
        n = self.__n
        x = self.__x
        y = self.__y
        if n >= self.__n_frames:
            self._atimer.Stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            self._atimer.Stop()
        PySide.QtCore.QCoreApplication.instance().processEvents()
        t_pause = scaled_dt - (time.time() - t0)
        if t_pause < 0:
            self.__f_skip += 1
            sys.stdout.flush()
        else:
            self._atimer.setInterval(t_pause * 1000.0)
        self.__n += self.__f_skip

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer') and self._atimer.IsRunning():
            self._atimer.Stop()
            return
        
        x, self.__y = self._qtwindow.current_data()
        self.__f_skip = 1
        self.__x = x[0]
        dt = self.__x[1] - self.__x[0]
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        print self.anim_time_scale * dt * 1000
        print self.__n_frames
        self._atimer = Timer( self.anim_time_scale * dt * 1000,
                              self.__step_frame )

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        v = View(
            VGroup(
                UItem('graph',
                      editor=CustomEditor(setup_qwidget_control,
                                          self._qtwindow.win)),
                HGroup(
                    VGroup(
                        Item('anim_time_scale', label='Divide real time'),
                        Item('anim_frame'),
                        label='Animate Frames'
                        ),
                    VGroup(
                        UItem('y_spacing'),
                        label='Trace offset (mV)'
                        ),
                    Group(
                        UItem('modules', style='custom',
                             editor=ListEditor(
                             use_notebook=True, deletable=False, 
                             dock_style='tab', page_name='.name',
                             ), height=-200
                        ),
                        show_border=True,
                        label='Modules'
                        )
                    ),
                ),
            width=1200,
            height=500,
            resizable=True,
            title='Quick Scanner'
        )
        return v
    
