import numpy as np
from scipy.linalg import LinAlgError
from scipy.signal import lfilter, lfilter_zi, hilbert
from scipy.interpolate import interp1d
import h5py
from tqdm import tqdm
from ecogdata.util import input_as_2d
from ecogdata.util import nextpow2


def h5mean(array, axis, rowmask=(), start=0, stop=None):
    """Compute mean of a 2D HDF5 array in blocks"""

    shape = array.shape
    if axis < 0:
        axis += len(shape)
    if stop is None:
        stop = shape[1]
    if axis==1:
        if len(rowmask):
            mn_size = rowmask.sum()
        else:
            mn_size = shape[0]
    else:
        mn_size = shape[1 - axis]
    mn = np.zeros(mn_size, 'd')
    # For averaging in both dimensions, still iterate chunks in time
    # If averaging over channels:
    # * fill in the chunk averages along the way
    # If averaging over time
    # * accumulate the samples (scaled by 1/N)
    itr = H5Chunks(array, axis=1, slices=True)
    for n, sl in tqdm(enumerate(itr), desc='Computing mean', leave=True, total=itr.n_blocks):
        t_sl = sl[1]
        # just pass until t_sl.start < requested start < t_sl.stop
        if start >= t_sl.stop:
            print('Continuing')
            continue
        # now modify first good slice
        elif start > t_sl.start:
            t_sl = slice(start, t_sl.stop)
            sl = (sl[0], t_sl)
        # break loops if stop < t_sl.start
        if stop < t_sl.start:
            break
        # now modify lsat good slice
        elif stop < t_sl.stop:
            t_sl = slice(t_sl.start, stop)
            sl = (sl[0], t_sl)
        x_sl = array[sl]
        if len(rowmask):
            x_sl = x_sl[rowmask]
        
        if axis == 0:
            mn[sl[1]] = x_sl.mean(0)
        else:
            mn[:] += x_sl.sum(1) / float(array.shape[1])
    return mn


def h5stat(array, fn, rowmask=()):
    """Compute timeseries of a channel-wise statistic for a 2D HDF5 array in blocks"""

    shape = array.shape
    T = shape[1]
    series = np.zeros(T, 'd')
    itr = H5Chunks(array, axis=1, slices=True)
    for n, sl in tqdm(enumerate(itr), desc='Computing series',
                      leave=True, total=itr.n_blocks):
        x_sl = array[sl]
        if len(rowmask):
            x_sl = x_sl[rowmask]
        series[sl[1]] = fn(x_sl)
    return series


class ReadCache(object):
    # TODO -- enable catch for full slicing
    """
    Buffers row indexes from memmap or hdf5 file.

    For cases where array[0, m:n], array[1, m:n], array[2, m:n] are
    accessed sequentially, this object buffers the C x (n-m)
    submatrix before yielding individual rows.

    Access such as array[p:q, m:n] is handled by the underlying
    array's __getitem__ method.
    """
    
    def __init__(self, array):
        self._array = array
        self._current_slice = None
        self._current_seg = ()
        self.dtype = array.dtype
        self.shape = array.shape

    def __len__(self):
        return len(self._array)

    @property
    def file_array(self):
        return self._array

    def __getitem__(self, sl):
        indx, srange = sl
        # Only access diretly if the first part of the slice is also a slice.
        # In other cases, slice all first and then use numpy indexing
        if isinstance(indx, slice):
            return self._array[sl].copy()
        if self._current_slice != srange:
            all_sl = (slice(None), srange)
            self._current_seg = self._array[all_sl]
            self._current_slice = srange
        # always return the full range after slicing with possibly
        # complex original range
        new_range = slice(None)
        new_sl = (indx, new_range)
        return self._current_seg[new_sl].copy()


class CommonReferenceReadCache(ReadCache):
    """Returns common-average re-referenced blocks"""

    def __getitem__(self, sl):
        indx, srange = sl
        if isinstance(indx, slice):
            # This returns without CAR?
            return self._array[sl].copy()
        if self._current_slice != srange:
            all_sl = (slice(None), srange)
            if self.dtype in np.sctypes['int']:
                self._current_seg = self._array[all_sl].astype('d')
            else:
                self._current_seg = self._array[all_sl].copy()
            self._current_seg -= self._current_seg.mean(0)
            self._current_slice = srange
        # always return the full range after slicing with possibly
        # complex original range
        new_range = slice(None)
        new_sl = (indx, new_range)
        return self._current_seg[new_sl].copy()


class FilteredReadCache(ReadCache):
    """
    Apply row-by-row filters to a ReadCache
    """

    def __init__(self, array, filters):
        if not isinstance(filters, (tuple, list)):
            f = filters
            filters = [ f ] * len(array)
        self.filters = filters
        super(FilteredReadCache, self).__init__(array)

    def __getitem__(self, sl):
        idx = sl[0]
        x = super(FilteredReadCache, self).__getitem__(sl)
        if isinstance(idx, int):
            return self.filters[idx]( x )
        y = np.empty_like(x)
        for x_, y_, f in zip(x[idx], y[idx], self.filters[idx]):
            y_[:] = f(x_)
        return y


def _make_subtract(z):
    def _f(x):
        return x - z
    return _f
    

class DCOffsetReadCache(FilteredReadCache):
    """
    A filtered read cache with a simple offset subtraction.
    """

    def __init__(self, array, offsets):
        #filters = [lambda x: x - off for off in offsets]
        filters = [_make_subtract(off) for off in offsets]
        super(DCOffsetReadCache, self).__init__(array, filters)
        self.offsets = offsets


class H5Chunks(object):
    """Iterates an HDF5 over "chunks" with ndarray-like access"""

    def __init__(self, h5array, out=None, axis=1, min_chunk=None, slices=False, reverse=False):
        """
        Efficient block iterator for HDF5 arrays (streams over chunking sizes to read whole blocks at a time).

        Parameters
        ----------
        h5array: h5py.Dataset
            Vector timeseries (chan x time) or (time x chan)
        out: h5py.Dataset
            Output array for write-back. May be equal to h5array for read/write arrays. Write-back disabled if None
        axis: int
            Axis to iterate over
        min_chunk: int
            Ensure the output blocks are greater than this size
        slices: bool
            Return array slicing rather than data
        reverse: bool
            Yield reverse-sequence data

        """
        chunk = h5array.chunks
        if len(chunk) > 2:
            raise ValueError('Only iterates for 2D arrays')
        self.h5array = h5array
        while axis < 0:
            axis += len(chunk)
        if chunk[axis] < chunk[1-axis]:
            print('chunk size larger in other dimension!')

        self.axis = axis
        self.size = h5array.shape[axis]
        self.chunk = chunk[axis]
        if min_chunk is not None:
            while self.chunk < min_chunk:
                self.chunk += chunk[axis]
        self.n_blocks = self.size // self.chunk
        if self.n_blocks * self.chunk < self.size:
            self.n_blocks += 1
        self.__it = self.n_blocks - 1 if reverse else 0
        self.reverse = reverse
        self.slices = slices
        self._output_source = out

    def write_out(self, data):
        if self._output_source is None:
            print('No output defined!')
            return
        if self.reverse:
            # data is reversed
            data = data[:, ::-1] if self.axis == 1 else data[::-1, :]
        self._output_source[self._current_sl] = data

    def __iter__(self):
        return self

    def __next__(self):
        if self.__it >= self.n_blocks or self.__it < 0:
            raise StopIteration()
        n = self.__it
        rng = slice(n * self.chunk, min(self.size, (n + 1) * self.chunk))
        self._current_sl = (slice(None), rng) if self.axis else (rng, slice(None))
        if self.reverse:
            self.__it -= 1
        else:
            self.__it += 1
        if self.slices:
            return self._current_sl
        arr = self.h5array[self._current_sl]
        if self.reverse:
            return arr[:, ::-1] if self.axis == 1 else arr[::-1, :]
        return arr


class HandOffIter:
    """
    Iterates over several 2D HDF5 arrays with hand-off between files. Hand-off procedure includes attemping to match
    the DC offsets between signals around the end and beginning of recording edges.

    Presently iterates over axis=1.

    Also supports write-back to the currently visible buffer within an iteration.

    """

    # TODO: support reverse iteration

    def __init__(self, arrays, out=None, min_chunk=None, chans=None, blank_points=10):
        """
        Construct hand-off iterator from HDF5 files.

        Parameters
        ----------
        arrays: sequence
            sequence of h5py.Datasets
        out: h5py.Dataset, str
            out may be a pre-created Dataset of the correct size or the path of output file. If output_file=='same',
            then write-back to the same input files. If None, then there is no output source.
        min_chunk: int
            Ensure the output blocks are greater than this size
        chans: list
            channels to expose on iteration (all by default)
        blank_points: int
            Blank these many points when handing off between files. Fill in +/- blank region with linear
            interpolation between valid points.

        """
        hdf_files = [array.file.filename for array in arrays]
        self.files = hdf_files
        self.arrays = arrays
        rec_lengths = [array.shape[1] for array in arrays]
        chunk_sizes = []
        num_blocks = 0
        if min_chunk is None:
            # todo: fix dumb 2000 pts hard coding
            min_chunk = blank_points + 2000
        else:
            min_chunk = max(blank_points + 2000, min_chunk)
        for array in arrays:
            size = array.chunks[1]
            if min_chunk is not None:
                while size < min_chunk:
                    size += array.chunks[1]
            if size > array.shape[1]:
                raise ValueError('Minimum chunk size {} is greater than the length of >=1 arrays'.format(min_chunk))
            chunk_sizes.append(size)
            # todo: is this +1 count correct?
            num_blocks += array.shape[1] // size + 1
        n_chan = arrays[0].shape[0]
        self.n_blocks = num_blocks
        if chans is None:
            chans = slice(None)
        else:
            if not np.iterable(chans):
                chans = (chans,)
            n_chan = len(chans)
        self.total_length = np.sum(rec_lengths)
        self.rec_lengths = rec_lengths
        self.chunk_sizes = chunk_sizes
        # Output status will be checked through the value of self._output_file:
        # if None, do nothing
        # if 'same', write back to input sources
        # else write to self._output_source defined here
        if isinstance(out, str):
            self._output_file = out
            if self._output_file.lower() != 'same':
                hdf = h5py.File(self._output_file, 'w')
                array_name = arrays[0].name.strip('/')
                out = hdf.create_dataset(array_name, shape=(n_chan, self.total_length), dtype='f', chunks=True)
                hdf.create_dataset('break_points', data=np.cumsum(rec_lengths[:-1], dtype='i'))
                self._output_source = out
                self._closing_output = True
        elif out is not None:
            self._output_source = out
            self._output_file = out.file.filename
            self._closing_output = False
        else:
            self._output_file = None
            self._closing_output = False
        self.chans = chans
        self._current_source = 0
        self._current_offset = None
        self._blanking_slice = False
        self._blank_points = blank_points

    def __iter__(self):
        # set up initial offset as the mean(s) in the first file
        self._current_source = self.arrays[0]
        means = self._slice_source(np.s_[self._blank_points:self._blank_points + 2000], offset=False).mean(axis=1)
        if self._output_file == 'same':
            self._output_source = self.arrays[0]
        self._current_source_num = 0
        self._current_offset = means[:, None]
        self._current_step = self.chunk_sizes[0]
        self._input_point = 0
        self._output_point = 0
        # starting on a blanking slice
        self._blanking_slice = True
        self._end_of_iter = False
        return self

    def _slice_source(self, time_slice, offset=True):
        if isinstance(self.chans, slice):
            arr = self._current_source[self.chans, time_slice]
        else:
            arr = np.array([self._current_source[c, time_slice] for c in self.chans])
        return arr - self._current_offset if offset else arr

    def _hand_off(self, start):
        # Right now the current step size will run off the end of the current source.
        # So grab the remainder of this source and hand-off to the next source.
        # Also reset the offset level to the average of the last few points
        # array_name = self.array_name
        end_point = self._current_source.shape[1]
        remainder = self._slice_source(np.s_[start:])
        old_mean = remainder.mean(1)[:, None]
        # Actually... use more points if the remainder is short
        if self._current_source.shape[1] - start < 100:
            longer_tail = self._slice_source(np.s[-100:])
            old_mean = longer_tail.mean(1)[:, None]
        # self._current_source.file.close()
        self._current_source_num += 1
        if self._current_source_num >= len(self.files):
            # do not change source or step size, just signal that the end is nigh
            self._end_of_iter = True
        else:
            self._current_source = self.arrays[self._current_source_num]
            self._current_step = self.chunk_sizes[self._current_source_num]
            self._blanking_slice = True
            self._break_point = self._output_point + (end_point - start)
            # get the mean of the first few points in the new source
            new_mean = self._slice_source(np.s_[self._blank_points:self._blank_points + 2000], offset=False).mean(1)
            # new_mean = np.array([self._current_source[c, self._blank_points:200].mean() for c in self.chans])
            # this is the offset to move the new mean to the old mean
            self._current_offset = new_mean[:, None] - old_mean
        return remainder

    def write_out(self, data):
        if self._output_file is None:
            print('No output file defined!')
            return
        elif self._output_file == 'same':
            # this condition means that data came from two sources in a hand-off
            if data.shape[1] > self._input_point:
                # last part is from current source
                self._current_source[:, :self._input_point] = data[:, -self._input_point:]
                # first part is from previous source
                n_prev = data.shape[1] - self._input_point
                prev_source = self.arrays[self._current_source_num - 1]
                prev_source[:, -n_prev:] = data[:, :n_prev]
            else:
                max_n = self._current_source.shape[1]
                start_pt = self._input_point - self._current_step
                stop_pt = min(max_n, self._input_point)
                this_slice = np.s_[:, start_pt:stop_pt]
                self._current_source[this_slice] = data
            return
        # Write this data into the output array.
        # If this is a blanking slice (b/c of hand-off) then ???
        a = self._output_point
        b = a + data.shape[1]
        self._output_source[:, a:b] = data
        self._output_source.flush()
        self._output_point = b


    def __next__(self):
        if self._end_of_iter:
            if self._closing_output:
                self._output_source.file.close()
            raise StopIteration
        start = self._input_point
        stop = start + self._current_step
        if stop > self._current_source.shape[1]:
            # print('hand off slice: {}-{}, file length {}'.format(start, stop, self._current_source.shape[1]))
            remainder = self._hand_off(start)
            # if the hand-off logic has found end-of-files then simply return the last bit and raise StopIteration
            # next time around
            if self._end_of_iter:
                # advance the input array point counter so that it can be rewound as needed in write_out
                self._input_point += self._current_step
                return remainder
            next_strip = self._slice_source(np.s_[:self._current_step])
            # Need to handle blanking!
            r_weight = np.linspace(0, 1, self._blank_points)
            left_point = remainder[:, -1][:, None]
            right_point = next_strip[:, self._blank_points][:, None]
            next_strip[:, :self._blank_points] = r_weight * right_point + (1 - r_weight) * left_point
            arr_slice = np.c_[remainder, next_strip]
            # next input is 1X the current step
            self._input_point = self._current_step
            # print('new input point: {}, file length {}'.format(self._input_point, self._current_source.shape[1]))
            return arr_slice
        else:
            # easy case!
            arr_slice = self._slice_source(np.s_[start:stop])
            self._input_point += self._current_step
            if start == 0 and self._current_source_num == 0:
                # just blank the initial points to zero
                arr_slice[:, :self._blank_points] = 0
            return arr_slice


def block_itr_factory(x, **kwargs):
    if isinstance(x, (tuple, list)):
        if 'axis' in kwargs and kwargs['axis'] == 1:
            # just drop this since it works right anyway
            kwargs.pop('axis')
        args = set(kwargs.keys())
        extra_args = args - {'out', 'min_chunks', 'chans', 'blank_points'}
        if len(extra_args):
            print('Dropping arguments not (yet) supported for HandOffIter: {}'.format(extra_args))
            supported_args = args - extra_args
            kwargs = dict((k, kwargs[k]) for k in supported_args)
        return HandOffIter(x, **kwargs)
    else:
        return H5Chunks(x, **kwargs)


def bfilter(b, a, x, axis=-1, out=None, filtfilt=False):
    """
    Apply linear filter inplace over array x streaming from disk.

    Parameters
    ----------
    b: ndarray
        polynomial coefs for transfer function denominator
    a: ndarray
        polynomial coefs for transfer function numerator
    x: h5py.Dataset, list
        Either a single or multiple datasets. If multiple, then a HandOffIter will be used to iterate. In this mode,
        if out is given as a string then the full output will be concatenated to a single HDF5 file. Otherwise output
        will be written back to each individual file.
    axis: int
        Array axis to apply filter
    out: h5py.Dataset, str
        Output array (or file name, see details above). If multiple inputs are given, a value of None will be
        converted to 'same'
    filtfilt: bool
        If True, perform zero-phase filtering with the forward-reverse technique

    Returns
    -------
    out: h5py.Dataset
        Output array. Not well defined if using HandOffIter in 'same' output mode

    """
    try:
        zii = lfilter_zi(b, a)
    except LinAlgError:
        # the integrating filter doesn't have valid zi
        zii = np.array([0.0])

    zi_sl = np.s_[None, :] if axis in (-1, 1) else np.s_[:, None]
    xc_sl = np.s_[:, :1] if axis in (-1, 1) else np.s_[:1, :]
    fir_size = len(b)
    if out is None:
        if isinstance(x, (list, tuple)):
            out = 'same'
        else:
            out = x
    itr = block_itr_factory(x, axis=axis, out=out, min_chunk=fir_size)
    for n, xc in tqdm(enumerate(itr), desc='Blockwise filtering',
                      leave=True, total=itr.n_blocks):
        if n == 0:
            zi = zii[zi_sl] * xc[xc_sl]
        xcf, zi = lfilter(b, a, xc, axis=axis, zi=zi)
        itr.write_out(xcf)

    # presently hand off iteration only goes forward so can't filt-filt
    if isinstance(itr, HandOffIter) or not filtfilt:
        out = itr._output_source
        del xc
        del xcf
        return out

    # Now read and write to the same out array (however it was earlier defined)
    itr = H5Chunks(out, axis=axis, min_chunk=fir_size, out=out, reverse=True)
    for n, xc in tqdm(enumerate(itr), desc='Blockwise filtering (reverse)',
                      leave=True, total=itr.n_blocks):
        if n == 0:
            zi = zii[zi_sl] * xc[xc_sl]
        xcf, zi = lfilter(b, a, xc, axis=axis, zi=zi)
        itr.write_out(xcf)
    del xc
    del xcf
    return out


def passthrough(x, y):
    itr = block_itr_factory(x, axis=1, out=y)
    for xc in tqdm(itr, desc='Copying to output', leave=True, total=itr.n_blocks):
        itr.write_out(xc)


@input_as_2d(in_arr=(0, 1))
def interpolate_blanked(x, mask, inplace=False, kind='linear'):
    if inplace:
        y = x
    else:
        y = x.copy()
    a = np.arange(x.shape[1])
    for row_x, row_y, row_m in zip(x, y, mask):
        fv = row_x[~row_m].mean()
        f = interp1d(a[~row_m], row_x[~row_m], kind=kind,
                     bounds_error=False, fill_value=fv)
        #row_y[~row_m] = row_x[~row_m]
        row_y[row_m] = f( a[row_m] )
    return y
    

def block_nan_filter(x, y, kind='linear'):
    itr = block_itr_factory(x, axis=1, out=y)
    for xc in tqdm(itr, desc='NaN Filtering', leave=True, total=itr.n_blocks):
        # xc = x[sl]
        nan_mask = np.isnan(xc)
        if not nan_mask.any():
            # y[sl] = xc
            itr.write_out(xc)
            continue
        xc = interpolate_blanked(xc, nan_mask, inplace=True, kind=kind)
        # y[sl] = xc
        itr.write_out(xc)
        

def square_filter(x, y):
    itr = block_itr_factory(x, axis=1, out=y)
    for xc in tqdm(itr, desc='Squaring', leave=True, total=itr.n_blocks):
        # xc = x[sl]
        # y[sl] = xc ** 2
        itr.write_out(xc ** 2)


def abs_filter(x, y):
    itr = block_itr_factory(x, axis=1, out=y)
    for xc in tqdm(itr, desc='Rectifying', leave=True, total=itr.n_blocks):
        # xc = x[sl]
        # y[sl] = np.abs(xc)
        itr.write_out(np.abs(xc))


def hilbert_envelope_filter(x, y):
    itr = block_itr_factory(x, axis=1, out=y)
    for xc in tqdm(itr, desc='Hilbert Transform', leave=True, total=itr.n_blocks):
        # xc = x[sl]
        n = xc.shape[1]
        nfft = nextpow2(n)

        # if n is closer to the previous power of 2, then split this block into two computations
        if (nfft - n) > (n - nfft / 2):
            n1 = int(n / 2)
            nfft = int(nfft / 2)
            y1 = hilbert(xc[..., :n1], N=nfft)[..., :n1]
            y2 = hilbert(xc[..., n1:], N=nfft)[..., :n - n1]
            # y[sl] = np.hstack((np.abs(y1), np.abs(y2)))
            itr.write_out(np.hstack((np.abs(y1), np.abs(y2))))
        else:
            y1 = hilbert(xc, N=nfft)[..., :n]
            # y[sl] = np.abs(y1)
            itr.write_out(np.abs(y1))
