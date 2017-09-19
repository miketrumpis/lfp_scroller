from __future__ import division
from tempfile import NamedTemporaryFile
import h5py

from traits.api import Instance, Button, HasTraits, Float, \
     List, Enum, Int, Bool
from traitsui.api import View, VGroup, Item, UItem, \
     Label, Handler, Group, TableEditor
from traitsui.table_column import ObjectColumn

from ecoglib.filt.time.design import butter_bp, cheby1_bp, cheby2_bp, \
     notch, ellip_bp, savgol
from ecoglib.filt.time import ma_highpass

from .h5data import bfilter

class FilterMenu(HasTraits):
    """Edits filter arguments."""
    filtfilt = Bool(False, info_text='Forward/reverse filter')
    def default_traits_view(self):
        # builds a panel automatically from traits
        t = self.traits()
        items = list()
        for k, v in t.items():
            if k in ('trait_added', 'trait_modified'):
                continue
            if v.info_text:
                items.extend( [Label(v.info_text), UItem(k, style='simple')] )
            else:
                items.extend( [UItem(k, style='simple')] )
        return View(*items, resizable=True)
    
class BandpassMenu(FilterMenu):
    lo = Float(-1, info_text='low freq cutoff (-1 for none)')
    hi = Float(-1, info_text='high freq cutoff (-1 for none)')
    ord = Int(5, info_text='filter order')

class ButterMenu(BandpassMenu):
    def make_filter(self, Fs):
        return butter_bp(lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord)

class Cheby1Menu(BandpassMenu):
    ripple = Float(0.1, info_text='passband ripple (dB)')
    def make_filter(self, Fs):
        return cheby1_bp(
            self.ripple, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
            )

class Cheby2Menu(BandpassMenu):
    stop_atten = Float(40, info_text='stopband attenuation (dB)')
    def make_filter(self, Fs):
        return cheby2_bp(
            self.stop_atten, lo=self.lo, hi=self.hi, Fs=Fs, ord=self.ord
            )
    
class NotchMenu(FilterMenu):
    notch = Float(60, info_text='notch frequency')
    nwid = Float(3.0, info_text='notch width (Hz)')
    nzo = Int(3, info_text='number of zeros at notch')
    def make_filter(self, Fs):
        return notch(self.notch, Fs, self.nwid, nzo=self.nzo)

class MAHPMenu(FilterMenu):
    hpc = Float(1, info_text='highpass cutoff')
    def make_filter(self, Fs):
        fir = ma_highpass(None, self.hpc/Fs, fir_filt=True)
        return fir, 1

class EllipMenu(BandpassMenu):
    atten = Float(10, info_text='Stopband attenutation (dB)')
    ripple = Float(0.5, info_text='Passband ripple (dB)')
    hp_width = Float(0, info_text='HP transition width (Hz)')
    lp_width = Float(0, info_text='LP transition width (Hz)')
    ord = Int(info_text='filter order')
    samp_rate = Float(2.0, info_text='Sampling rate')
    check_order = Button('check order')

    def _check_order_fired(self):
        b, a = self.make_filter(self.samp_rate)
        self.ord = len(b)

    def make_filter(self, Fs):
        return ellip_bp(
            self.atten, self.ripple, lo=self.lo, hi=self.hi,
            hp_width=self.hp_width, lp_width=self.lp_width, Fs=Fs
            )

class SavgolMenu(FilterMenu):
    T = Float(1.0, info_text='Window length (s)')
    ord = Int(3, info_text='Poly order')
    sm = Bool(True, info_text='Smoothing (T), residual (F)')

    def make_filter(self, Fs):
        return savgol(self.T, self.ord, Fs=Fs, smoothing=self.sm)
    
class FilterHandler(Handler):

    def object_filt_type_changed(self, info):
        # check whether there is already a menu of the
        # correct type -- this method appears to be called
        # whenever a new handler is created!
        if info.object.filt_type == 'butterworth':
            if not isinstance(info.object.filt_menu, ButterMenu):
                info.object.filt_menu = ButterMenu()
        elif info.object.filt_type == 'cheby 1':
            if not isinstance(info.object.filt_menu, Cheby1Menu):
                info.object.filt_menu = Cheby1Menu()
        elif info.object.filt_type == 'cheby 2':
            if not isinstance(info.object.filt_menu, Cheby2Menu):
                info.object.filt_menu = Cheby2Menu()
        elif info.object.filt_type == 'elliptical':
            if not isinstance(info.object.filt_menu, EllipMenu):
                info.object.filt_menu = EllipMenu()            
        elif info.object.filt_type == 'notch':
            if not isinstance(info.object.filt_menu, NotchMenu):
                info.object.filt_menu = NotchMenu()
        elif info.object.filt_type == 'm-avg hp':
            if not isinstance(info.object.filt_menu, MAHPMenu):
                info.object.filt_menu = MAHPMenu()
        elif info.object.filt_type == 'savitzky-golay':
            if not isinstance(info.object.filt_menu, SavgolMenu):
                info.object.filt_menu = SavgolMenu()
            

available_filters = ('butterworth',
                     'cheby 1',
                     'cheby 2',
                     'elliptical',
                     'notch',
                     'm-avg hp',
                     'savitzky-golay')

class Filter(HasTraits):
    """Filter model with adaptive menu. Menu can build transfer functions."""
    
    filt_type = Enum( 'butterworth', available_filters )
    filt_menu = Instance(HasTraits)

    view = View(
        VGroup(
            Item('filt_type', label='Filter type'),
            Group(
                UItem('filt_menu', style='custom'),
                label='Filter menu'
                )
            ),
        handler=FilterHandler,
        resizable=True
        )
    
filter_table = TableEditor(
    columns = [ ObjectColumn( name='filt_type' ) ],
    deletable=True,
    auto_size=True,
    show_toolbar=True,
    reorderable=True,
    edit_view=None,
    row_factory=Filter,
    selected='selected'
    )

def pipeline_factory(f_list, filter_modes):
    if not len(f_list):
        return lambda x: x
    def copy_x(x):
        with NamedTemporaryFile(mode='ab', dir='.') as f:
            f.file.close()
            fw = h5py.File(f.name, 'w')
            y = fw.create_dataset('data', shape=x.shape,
                                  dtype=x.dtype, chunks=x.chunks)
        return y
        
    def run_list(x, axis=1):
        y = copy_x(x)
        b, a = f_list[0]
        bfilter(b, a, x, axis=axis, out=y, filtfilt=filter_modes[0])
        for (b, a), ff in zip(f_list[1:], filter_modes[1:]):
            bfilter(b, a, y, axis=axis, filtfilt=ff)
        return y

    return run_list
        

class FilterPipeline(HasTraits):
    """List of filters that can be pipelined."""
    filters = List(Filter)
    filt_type = Enum( 'butterworth', available_filters )
    selected = Instance(Filter)
    add_filter = Button('Add filter')
    remove_filter = Button('Remove filter')

    def make_pipeline(self, Fs):
        filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
        filtfilt = [f.filt_menu.filtfilt for f in self.filters]
        return pipeline_factory(filters, filtfilt)

    def _add_filter_fired(self):
        self.filters.append( Filter(filt_type=self.filt_type) )

    def _remove_filter_fired(self):
        self.filters.remove( self.selected )
    
    view = View(
        VGroup(
            Group(
                UItem('filt_type'),
                UItem('add_filter'),
                UItem('remove_filter'),
                springy=True
                ),
            Group(
                UItem('filters', editor=filter_table),
                show_border=True
                ),
            ),
        resizable=True,
        title='Filters',
        )
