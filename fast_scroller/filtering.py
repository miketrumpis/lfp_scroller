from tempfile import NamedTemporaryFile
from functools import partial
import h5py
import pickle
import json
import base64

from traits.api import Instance, Button, HasTraits, Float, \
     List, Enum, Int, Bool
from traitsui.api import View, VGroup, HGroup, Item, UItem, \
     Label, Handler, Group, TableEditor
from traitsui.table_column import ObjectColumn
from pyface.api import FileDialog, OK

from ecogdata.filt.time.design import butter_bp, cheby1_bp, cheby2_bp, \
     notch, ellip_bp, savgol
from ecogdata.filt.time import ma_highpass

from .h5data import bfilter, block_nan_filter, square_filter, abs_filter, hilbert_envelope_filter


class ReportsTraits(HasTraits):
    """Creates a dictionary of its own traits values"""

    def to_dict(self, values=True, private=False):
        t = self.traits()
        excluded = ['trait_added', 'trait_modified']
        if not private:
            excluded.extend([k for k in t.keys() if k.startswith('_')])
        if values:
            items = dict()
            for k in t.keys():
                if k in excluded:
                    continue
                items.update(self.trait_get(k))
        else:
            items = dict([(k, v) for k, v in t.items() if k not in excluded])
        return items


class AutoPanel(ReportsTraits):
    """Auto-populates a view panel"""
    orientation='vertical'

    def default_traits_view(self):
        # builds a panel automatically from traits
        items = list()
        for k, v in self.to_dict(values=False).items():
            if v.info_text:
                items.extend( [Label(v.info_text), UItem(k, style='simple')] )
            else:
                items.extend( [UItem(k, style='simple')] )
        if self.orientation == 'vertical':
            group = VGroup(*items)
        else:
            group = HGroup(*items)
        return View(group, resizable=True)


class FilterMenu(AutoPanel):
    """Edits filter arguments."""

    filtfilt = Bool(False, info_text='Forward/reverse filter')


class NaNFilterMenu(AutoPanel):
    """Interpolate through NaN code values"""

    interp_order = Enum('linear', ('linear', 'nearest', 'zero',
                                   'slinear', 'quadratic', 'cubic'))

    def make_filter(self, Fs):
        return partial(block_nan_filter, kind=self.interp_order)


class SquareFilterMenu(AutoPanel):
    """Square voltage values"""

    def make_filter(self, Fs):
        return square_filter


class AbsFilterMenu(AutoPanel):
    """Rectify voltage values"""

    def make_filter(self, Fs):
        return abs_filter


class HilbertEnvMenu(AutoPanel):
    """Rectify voltage values"""

    def make_filter(self, Fs):
        return hilbert_envelope_filter


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
        elif info.object.filt_type == 'NaN filter':
            if not isinstance(info.object.filt_menu, NaNFilterMenu):
                info.object.filt_menu = NaNFilterMenu()
        elif info.object.filt_type == 'square rectifier':
            if not isinstance(info.object.filt_menu, SquareFilterMenu):
                info.object.filt_menu = SquareFilterMenu()
        elif info.object.filt_type == 'abs rectifier':
            if not isinstance(info.object.filt_menu, AbsFilterMenu):
                info.object.filt_menu = AbsFilterMenu()
        elif info.object.filt_type == 'hilbert envelope':
            if not isinstance(info.object.filt_menu, HilbertEnvMenu):
                info.object.filt_menu = HilbertEnvMenu()


# These items match the name and case of filt_types in the switch statement in FilterHandler
available_filters = ('NaN filter',
                     'butterworth',
                     'cheby 1',
                     'cheby 2',
                     'elliptical',
                     'notch',
                     'm-avg hp',
                     'savitzky-golay',
                     'square rectifier',
                     'abs rectifier',
                     'hilbert envelope')


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


def pipeline_factory(f_list, filter_modes, output_file=None, data_field='data'):
    # TODO: if f_list is empty, is it consistent not to create an output file?
    if not len(f_list):
        return lambda x: x

    def get_x_props(x):
        if isinstance(x, (list, tuple)):
            time = sum([arr.shape[1] for arr in x])
            chan = x[0].shape[0]
            shape = (chan, time)
            dtype = x[0].dtype
            # TODO: handle case for concatenating on axis=0
            return shape, dtype
        return x.shape, x.dtype

    def anonymous_copy_x(x):
        with NamedTemporaryFile(mode='ab', dir='.') as f:
            # This redundant f.file.close is needed by Windows before re-opening the file
            f.file.close()
            fw = h5py.File(f.name, 'w')
            shape, dtype = get_x_props(x)
            y = fw.create_dataset(data_field, shape=shape, dtype=dtype, chunks=True)
        return y

    def copy_x(x):
        fw = h5py.File(output_file, 'w')
        shape, dtype = get_x_props(x)
        y = fw.create_dataset(data_field, shape=shape, dtype=dtype, chunks=True)
        return y
        
    def run_list(x, axis=1):
        if output_file is not None:
            if output_file.lower() == 'same':
                y = 'same'
            else:
                y = copy_x(x)
        else:
            y = anonymous_copy_x(x)
        # call first filter to initialize y
        if callable(f_list[0]):
            nf = f_list[0]
            nf(x, y)
        else:
            b, a = f_list[0]
            bfilter(b, a, x, axis=axis, out=y, filtfilt=filter_modes[0])
        # cycle through the rest of the filters as y <- F(y)
        for filt, ff_mode in zip(f_list[1:], filter_modes[1:]):
            if callable(filt):
                filt(y, y)
            else:
                b, a = filt
                bfilter(b, a, y, axis=axis, filtfilt=ff_mode)
        return y

    return run_list
        

class FilterPipeline(HasTraits):
    """List of filters that can be pipelined."""
    filters = List(Filter)
    filt_type = Enum( 'butterworth', available_filters )
    selected = Instance(Filter)
    add_filter = Button('Add filter')
    remove_filter = Button('Remove filter')
    b_save_pipeline = Button('Save pipeline')
    b_load_pipeline = Button('Load pipeline')

    def __len__(self):
        return len(self.filters)

    def serialize_pipeline(self):
        d = dict()
        f_seq = []
        for f in self.filters:
            f_params = f.filt_menu.to_dict(values=True)
            f_params['filter type'] = f.filt_type
            f_seq.append(f_params)
        d['filters'] = f_seq
        # write the pickle bytes after some encode filtering (not at all understood)
        # from here:
        # https://www.pythonforthelab.com/blog/storing-binary-data-and-serializing/#combining-json-and-pickle
        d['filters_pickle'] = base64.b64encode(pickle.dumps(self.filters)).decode('ascii')
        return json.dumps(d, sort_keys=True, indent=2, separators=(',', ':'))


    def save_pipeline(self, path):
        pipe_str = self.serialize_pipeline()
        open(path, 'w').write(pipe_str)


    def load_pipeline(self, path):
        with open(path, 'r') as fp:
            pipe_info = json.load(fp)
        f_pickle = pipe_info['filters_pickle']
        # From https://www.pythonforthelab.com/blog/storing-binary-data-and-serializing/#combining-json-and-pickle
        # reverse the ascii encoding back to bytes from pickle
        pipeline = pickle.loads(base64.b64decode(f_pickle))
        self.filters = pipeline


    def _b_save_pipeline_fired(self):
        saver = FileDialog(action='save as')
        if saver.open() == OK:
            self.save_pipeline(saver.path)


    def _b_load_pipeline_fired(self):
        loader = FileDialog(title='Open file')
        if loader.open() == OK:
            self.load_pipeline(loader.path)


    def make_pipeline(self, Fs, **kwargs):
        filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
        filtfilt = []
        for f in self.filters:
            try:
                filtfilt.append(f.filt_menu.filtfilt)
            except AttributeError:
                filtfilt.append(None)
        return pipeline_factory(filters, filtfilt, **kwargs)


    def _add_filter_fired(self):
        self.filters.append( Filter(filt_type=self.filt_type) )


    def _remove_filter_fired(self):
        self.filters.remove( self.selected )
    

    view = View(
        VGroup(
            HGroup(
                Group(
                    UItem('filt_type'),
                    UItem('add_filter'),
                    UItem('remove_filter'),
                    springy=True
                    ),
                Group(
                    UItem('b_load_pipeline'),
                    UItem('b_save_pipeline'),
                    springy=True
                ),
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
