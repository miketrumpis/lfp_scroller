from tempfile import NamedTemporaryFile
from functools import partial
import h5py
import pickle
import json
import base64
import numpy as np

from traits.api import Instance, Button, HasTraits, Float, List, Enum, Int, Bool
from traitsui.api import View, VGroup, HGroup, Item, UItem, Label, Handler, Group, TableEditor
from traitsui.table_column import ObjectColumn
from pyface.api import FileDialog, OK

from ecogdata.filt.time.design import butter_bp, cheby1_bp, cheby2_bp, notch, ellip_bp, savgol
from ecogdata.filt.time import ma_highpass

from .h5data import bfilter, block_nan_filter, passthrough, square_filter, abs_filter, hilbert_envelope_filter


class ReportsTraits(HasTraits):
    """Creates a dictionary of its own traits values"""
    excluded_traits = ['trait_added', 'trait_modified']

    def to_dict(self, values=True, private=False):
        t = self.traits()
        excluded = self.excluded_traits[:]
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
    orientation = 'vertical'

    def default_traits_view(self):
        # builds a panel automatically from traits
        items = list()
        for k, v in self.to_dict(values=False).items():
            if v.info_text:
                items.extend([Label(v.info_text), UItem(k, style='simple')])
            else:
                items.extend([UItem(k, style='simple')])
        if self.orientation == 'vertical':
            group = VGroup(*items)
        else:
            group = HGroup(*items)
        return View(group, resizable=True)


class FilterMenu(AutoPanel):
    """Edits filter arguments."""

    def validate_params(self, Fs):
        return True


class NaNFilterMenu(FilterMenu):
    """Interpolate through NaN code values"""

    interp_order = Enum('linear', ('linear', 'nearest', 'zero',
                                   'slinear', 'quadratic', 'cubic'))

    def make_filter(self, Fs):
        return partial(block_nan_filter, kind=self.interp_order)


class SquareFilterMenu(FilterMenu):
    """Square voltage values"""

    def make_filter(self, Fs):
        return square_filter


class AbsFilterMenu(FilterMenu):
    """Rectify voltage values"""

    def make_filter(self, Fs):
        return abs_filter


class HilbertEnvMenu(FilterMenu):
    """Rectify voltage values"""

    def make_filter(self, Fs):
        return hilbert_envelope_filter


class LTIFilterMenu(FilterMenu):
    """LTI filter arguments."""

    filtfilt = Bool(False, info_text='Forward/reverse filter')


class BandpassMenu(LTIFilterMenu):
    lo = Float(-1, info_text='low freq cutoff (-1 for none)')
    hi = Float(-1, info_text='high freq cutoff (-1 for none)')
    ord = Int(5, info_text='filter order')

    def validate_params(self, Fs):
        edge_set = self.lo > 0 or self.hi > 0
        edge_valid = self.lo < Fs and self.hi < Fs
        return edge_set and edge_valid


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


class NotchMenu(LTIFilterMenu):
    notch = Float(60, info_text='notch frequency')
    nwid = Float(3.0, info_text='notch width (Hz)')
    nzo = Int(3, info_text='number of zeros at notch')

    def make_filter(self, Fs):
        return notch(self.notch, Fs, self.nwid, nzo=self.nzo)

    def validate_params(self, Fs):
        notch_valid = self.notch < Fs
        return notch_valid


class MAHPMenu(LTIFilterMenu):
    hpc = Float(1, info_text='highpass cutoff')

    def make_filter(self, Fs):
        fir = ma_highpass(None, self.hpc/Fs, fir_filt=True)
        return fir, 1

    def validate_params(self, Fs):
        return self.hpc > 0 and self.hpc < Fs


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

    def validate_params(self, Fs):
        valid_edges = super().validate_params(Fs)
        trans_valid = True
        if self.lo > 0:
            trans_valid = trans_valid and self.lp_width > 0
        if self.hi > 0:
            trans_valid = trans_valid and self.hp_width > 0
        return valid_edges and trans_valid



class SavgolMenu(LTIFilterMenu):
    T = Float(1.0, info_text='Window length (s)')
    ord = Int(3, info_text='Poly order')
    sm = Bool(True, info_text='Smoothing (T), residual (F)')

    def make_filter(self, Fs):
        return savgol(self.T, self.ord, Fs=Fs, smoothing=self.sm)

    def validate_params(self, Fs):
        return self.T > 0 and self.ord > 0


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

    filt_type = Enum('butterworth', available_filters)
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
    columns=[ObjectColumn(name='filt_type')],
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


class PipelineFactory:

    def __init__(self, filter_list: list, filter_modes: list, transpose_input: bool=False,
                 output_file: str=None, data_field: str='data'):
        """

        Parameters
        ----------
        filter_list : list
            List of filters that are either polynomial coefficients, or callable functions
            with signature filt(input, output)
        filter_modes : list
            List of booleans that indicate filtfilt mode (n/a for callables,
            but there must still be a placeholder)
        transpose_input: bool
            If True, do a first step to transpose to input to channels x time orientation
        output_file : str
            If given, the output file name
        data_field : str
            If given, the name of the data array in the output file

        """
        self.filter_list = filter_list
        self.filter_modes = filter_modes
        self.output_file = output_file
        self.data_field = data_field
        self.transpose = transpose_input

    def get_x_props(self, x):
        if isinstance(x, (list, tuple)):
            time = sum([arr.shape[1] for arr in x])
            chan = x[0].shape[0]
            shape = (chan, time)
            dtype = x[0].dtype
            # TODO: handle case for concatenating on axis=0
            return shape, dtype
        return x.shape, x.dtype

    def anonymous_copy_x(self, x, integer_ok: bool=False):
        with NamedTemporaryFile(mode='ab', dir='.') as f:
            # This redundant f.file.close is needed by Windows before re-opening the file
            f.file.close()
            fw = h5py.File(f.name, 'w')
            shape, dtype = self.get_x_props(x)
            if self.transpose:
                shape = shape[::-1]
            if dtype in np.sctypes['int'] + np.sctypes['uint'] and not integer_ok:
                dtype = 'f'
            y = fw.create_dataset(self.data_field, shape=shape, dtype=dtype, chunks=True)
        return y

    def copy_x(self, x, integer_ok: bool=False):
        fw = h5py.File(self.output_file, 'w')
        shape, dtype = self.get_x_props(x)
        if self.transpose:
            shape = shape[::-1]
        if dtype in np.sctypes['int'] + np.sctypes['uint'] and not integer_ok:
            dtype = 'f'
        y = fw.create_dataset(self.data_field, shape=shape, dtype=dtype, chunks=True)
        return y

    @property
    def filter_pipeline(self):
        return self._filter_pipeline

    def _filter_pipeline(self, x, axis=1):
        # simply return x if no filtering or copying is needed
        if not len(self.filter_list):
            if not self.transpose:
                return x
            else:
                integer_ok = True
        else:
            integer_ok = False

        print(f'Preparing output HDF with integer {integer_ok}')
        if self.output_file is not None:
            if self.output_file.lower() == 'same':
                y = 'same'
            else:
                y = self.copy_x(x, integer_ok=integer_ok)
        else:
            y = self.anonymous_copy_x(x, integer_ok=integer_ok)
        # do a transpose passthrough if needed
        if self.transpose:
            passthrough(x, y, transpose=True)
            x = y
        # can return the transpose if there is no other filtering
        if not len(self.filter_list):
            return x
        # call first filter to initialize y
        if callable(self.filter_list[0]):
            nf = self.filter_list[0]
            nf(x, y)
        else:
            b, a = self.filter_list[0]
            bfilter(b, a, x, axis=axis, out=y, filtfilt=self.filter_modes[0])
        # cycle through the rest of the filters as y <- F(y)
        for filt, ff_mode in zip(self.filter_list[1:], self.filter_modes[1:]):
            if callable(filt):
                filt(y, y)
            else:
                b, a = filt
                bfilter(b, a, y, axis=axis, filtfilt=ff_mode)
        return y


class FilterPipeline(HasTraits):
    """List of filters that can be pipelined."""
    filters = List(Filter)
    filt_type = Enum('butterworth', available_filters)
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

    def make_pipeline(self, Fs, **kwargs) -> PipelineFactory:
        valid_filters = list()
        for f in self.filters:
            if f.filt_menu is None or not f.filt_menu.validate_params(Fs):
                continue
            valid_filters.append(f.filt_menu)
        # valid_filters = [f.filt_menu for f in self.filters if f.filt_menu.validate_params(Fs)]
        filters = [f.make_filter(Fs) for f in valid_filters]
        # filters = [f.filt_menu.make_filter(Fs) for f in self.filters]
        filtfilt = []
        for f in filters:
            try:
                filtfilt.append(f.filtfilt)
            except AttributeError:
                filtfilt.append(None)
        # return pipeline_factory(filters, filtfilt, **kwargs)
        return PipelineFactory(filters, filtfilt, **kwargs)

    def _add_filter_fired(self):
        self.filters.append(Filter(filt_type=self.filt_type))

    def _remove_filter_fired(self):
        self.filters.remove(self.selected)

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
