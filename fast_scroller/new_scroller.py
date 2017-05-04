from __future__ import division

import matplotlib as mpl
mpl.rcParams['interactive'] = True
mpl.use('Qt4Agg')
from matplotlib.figure import Figure
import sys
import numpy as np
import PySide
from pyqtgraph.Qt import QtGui

from traits.api import Instance, Button, HasTraits, Float, Bool, \
     Enum, on_trait_change, Str, List
from traitsui.api import View, VGroup, HGroup, Item, UItem, CustomEditor, \
     Group, ListEditor, VSplit, HSplit, Label

from pyface.timer.api import Timer
import time

from ecogana.anacode.anatool.gui_tools import SavesFigure
from ecogana.anacode.plot_util import filled_interval, subplots
from ecoglib.estimation.jackknife import Jackknife
from ecoglib.numutil import nextpow2
from sandbox.split_methods import multi_taper_psd

import ecogana.anacode.seaborn_lite as sns


## from matplotlib.backends.backend_qt4agg import \
##      FigureCanvasQTAgg as FigureCanvas
## from matplotlib.backends.backend_qt4agg import \
##      NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt4agg import new_figure_manager_given_figure
def simplefigure(fig):
    from pyqtgraph.Qt import QtCore
    canvas = FigureCanvas(fig)
    main_frame = QtGui.QWidget()
    canvas.setParent(main_frame)
    mpl_toolbar = NavigationToolbar(canvas, main_frame)        
    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(canvas)
    vbox.addWidget(mpl_toolbar)
    main_frame.setLayout(vbox)
    #main_frame.setGeometry(QtCore.QRect(100, 100, 400, 200))
    return main_frame



class VisModule(HasTraits):
    name = Str('__dummy___')
    parent = Instance('VisWrapper')

class PlotsInterval(VisModule):
    new_figure = Bool(False)
    _figures = List([])

    def _get_fig(self, **kwargs):
        """Accepts subplots kwargs"""
        fig = getattr(self, 'fig', None)
        if fig and not self.new_figure:
            return fig, self.axes, fig.canvas.draw_idle
        self.fig, self.axes = subplots(**kwargs)
        splot = SavesFigure.live_fig(self.fig)
        self._figures.append(splot)
        self._plot_count = 0
        self._colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        return self.fig, self.axes, self.fig.canvas.draw_idle

class IntervalTraces(PlotsInterval):
    name = 'Plot Window'
    plot = Button('Plot')
    new_figure = Bool(True)

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        x = x[0]
        dy = self.parent.y_spacing
        f, ax, update = self._get_fig()
        clr = self._colors[self._plot_count]
        ax.plot(x, y.T + np.arange(len(y)) * dy, lw=0.5, color=clr)
        sns.despine(ax=ax)
        ax.set_ylabel('Amplitude (mV)')
        ax.set_xlabel('Time (s)')
        f.tight_layout()
        self._plot_count += 1
        try:
            update()
        except:
            pass

    def default_traits_view(self):
        v = View(
            Group(
                UItem('plot'),
                label='Plot current window'
                )
            )
        return v

class IntervalSpectrum(PlotsInterval):
    name = 'Power Spectrum'
    NW = Enum( 2.5, np.arange(2, 11, 0.5).tolist() )
    pow2 = Bool(True)
    avg_spec = Bool(True)
    sem = Bool(True)
    adaptive = Bool(True)
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
            ## l_pxx = np.log(pxx)
            ## mn = l_pxx.mean(0)
            ## se = l_pxx.std(0) / np.sqrt(len(l_pxx))
            se = np.exp( mn - se ), np.exp( mn + se )
            filled_interval(
                ax.semilogy, fx, np.exp(mn), se,
                color=self._colors[self._plot_count], label=label, ax=ax
                )
            ax.legend()
        else:
            lns = ax.semilogy(
                fx, pxx.T, lw=.25, color=self._colors[self._plot_count]
                )
            self._plot_count += 1
            ax.legend(lns[:1], (label,))
        ax.set_ylabel('Power Spectral Density (mV / Hz^2)')
        ax.set_xlabel('Frequency (Hz)')
        sns.despine(ax=ax)
        fig.tight_layout()
        self._plot_count += 1
        if update is not None:
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

class YRange(VisModule):
    name = 'Trace Offsets'
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self.parent._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        v = View(
            Group(
                UItem('y_spacing'),
                label='Trace spacing (mv)'
                )
            )
        return v

class AnimateInterval(VisModule):
    name = 'Animate window'
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 20, 50, 100])

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
            self.parent._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            self._atimer.Stop()
        PySide.QtCore.QCoreApplication.instance().processEvents()
        # calculate the difference between the desired interval
        # and the time it just took to draw (per frame)
        t_pause = scaled_dt - (time.time() - t0) / self.__f_skip
        if t_pause < 0:
            self.__f_skip += 1
            print 'took', -t_pause, 'sec too long, increasing skip rate', self.__f_skip
            sys.stdout.flush()
        else:
            print t_pause
            self._atimer.setInterval(t_pause * 1000.0)
        self.__n += self.__f_skip

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer') and self._atimer.IsRunning():
            self._atimer.Stop()
            return
        
        x, self.__y = self.parent._qtwindow.current_data()
        self.__f_skip = 1
        self.__x = x[0]
        dt = self.__x[1] - self.__x[0]
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        print self.anim_time_scale * dt * 1000
        print self.__n_frames
        self._atimer = Timer( self.anim_time_scale * dt * 1000,
                              self.__step_frame )

    def default_traits_view(self):
        v = View(
            VGroup(
                Item('anim_time_scale', label='Divide real time'),
                Item('anim_frame'),
                label='Animate Frames'
                )
            )
        return v
    
def setup_qwidget_control(parent, editor, qwidget):
    return qwidget

class VisWrapper(HasTraits):

    graph = Instance(QtGui.QWidget)
    x_scale = Float
    y_spacing = Enum(0.5, [0.1, 0.2, 0.5, 1, 2, 5])

    modules = List( [#YRange,
                     IntervalTraces,
                     AnimateInterval,
                     IntervalSpectrum] )

    def __init__(self, qtwindow, **traits):
        self._qtwindow = qtwindow
        HasTraits.__init__(self, **traits)
        self._make_modules()

    def _make_modules(self):
        mods = self.modules[:]
        self.modules = []
        for m in mods:
            self.modules.append( m(parent=self) )

    @on_trait_change('y_spacing')
    def _change_spacing(self):
        self._qtwindow.update_y_spacing(self.y_spacing)

    def default_traits_view(self):
        ht = 1200
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
    
