from __future__ import division
import numpy as np
from scipy.linalg import LinAlgError
from scipy.signal import lfilter, lfilter_zi
from scipy.interpolate import interp1d
from tqdm import tqdm
from ecoglib.util import input_as_2d

def h5mean(
        array, axis, block_size=20000,
        start=0, stop=None, rowmask=(), colmask=()
        ):
    """Compute mean of a 2D HDF5 array in blocks"""

    shape = array.shape
    if axis < 0:
        axis += len(shape)
    mn_size = shape[1 - axis]
    mn = np.empty(mn_size, 'd')
    b = 0
    while True:
        range = slice( b, min(mn_size, b + block_size) )
        # to compute block x, either slice like:
        # array[:, x].mean(0)
        # array[x, :].mean(1)
        if axis == 1:
            sl = ( range, slice(start, stop) )
        else:
            sl = ( slice(start, stop), range )
        if not (len(rowmask) or len(colmask)):
            mn[range] = array[sl].mean(axis)
        else:
            tmp = array[sl]
            # only average unmasked rows/columns
            if len(rowmask):
                tmp = tmp[rowmask]
            if len(colmask):
                tmp = tmp[:, colmask]
            mn[range] = tmp.mean(axis)
        b += block_size
        if b >= mn_size:
            break
    return mn

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
        indx, range = sl
        if not isinstance(indx, (np.integer, int)):
            return self._array[sl].copy()
        if self._current_slice != range:
            all_sl = ( slice(None), range )
            self._current_seg = self._array[all_sl]
            self._current_slice = range
        # always return the full range after slicing with possibly
        # complex original range
        new_range = slice(None)
        new_sl = (indx, new_range)
        return self._current_seg[new_sl].copy()

class CommonReferenceReadCache(ReadCache):
    """Returns common-average re-referenced blocks"""

    def __getitem__(self, sl):
        indx, range = sl
        if not isinstance(indx, (np.integer, int)):
            return self._array[sl].copy()
        if self._current_slice != range:
            all_sl = ( slice(None), range )
            self._current_seg = self._array[all_sl].copy()
            self._current_seg -= self._current_seg.mean(0)
            self._current_slice = range
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

    def __init__(
            self, h5array, axis=1, min_chunk=None, slices=False, reverse=False
            ):
        chunk = h5array.chunks
        if len(chunk) > 2:
            raise ValueError('Only iterates for 2D arrays')
        self.h5array = h5array
        while axis < 0:
            axis += len(chunk)
        if chunk[axis] < chunk[1-axis]:
            print 'chunk size larger is other dimension!'

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

    def __iter__(self):
        return self

    def next(self):
        if self.__it >= self.n_blocks or self.__it < 0:
            raise StopIteration()
        n = self.__it
        rng = slice(n * self.chunk, min(self.size, (n+1) * self.chunk))
        sl = ( slice(None), rng ) if self.axis else ( rng, slice(None) )
        if self.reverse:
            self.__it -= 1
        else:
            self.__it += 1
        if self.slices:
            return sl
        return self.h5array[sl]

def bfilter(b, a, x, axis=-1, out=None, filtfilt=False):
    """
    Apply linear filter inplace over array x streaming from disk.
    """

    try:
        zii = lfilter_zi(b, a)
    except LinAlgError:
        # the integrating filter doesn't have valid zi
        zii = np.array([0.0])
        
    zi_sl = [np.newaxis] * x.ndim
    zi_sl[axis] = slice(None)
    xc_sl = [slice(None)] * x.ndim
    xc_sl[axis] = slice(0,1)
    fir_size = len(b)
    itr = H5Chunks(x, axis=axis, min_chunk=fir_size, slices=True)
    for n, sl in tqdm(enumerate(itr), desc='Blockwise filtering',
                      leave=True, total=itr.n_blocks):
        xc = x[sl]
        if n == 0:
            zi = zii[ tuple(zi_sl) ] * xc[ tuple(xc_sl) ]
        xcf, zi = lfilter(b, a, xc, axis=axis, zi=zi)
        if out is None:
            x[sl] = xcf
        else:
            out[sl] = xcf

    if not filtfilt:
        del xc
        del xcf
        return

    rev_sl = [ slice(None) ] * x.ndim
    rev_sl[axis] = slice(None, None, -1)
    itr = H5Chunks(x, axis=axis, min_chunk=fir_size, slices=True, reverse=True)
    for n, sl in tqdm(enumerate(itr), desc='Blockwise filtering (reverse)',
                      leave=True, total=itr.n_blocks):
        if out is None:
            xc = x[sl][rev_sl]
        else:
            xc = out[sl][rev_sl]
        if n == 0:
            zi = zii[ tuple(zi_sl) ] * xc[ tuple(xc_sl) ]
        xcf, zi = lfilter(b, a, xc, axis=axis, zi=zi)
        if out is None:
            x[sl] = xcf[rev_sl]
        else:
            out[sl] = xcf[rev_sl]
    del xc
    del xcf

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
    itr = H5Chunks(x, axis=1, slices=True)
    for n, sl in tqdm(enumerate(itr), desc='NaN Filtering',
                      leave=True, total=itr.n_blocks):
        xc = x[sl]
        nan_mask = np.isnan(xc)
        if not nan_mask.any():
            y[sl] = xc
            continue
        xc = interpolate_blanked(xc, nan_mask, inplace=True, kind=kind)
        y[sl] = xc
        
        ## yc = np.empty_like(xc)
        ## nm = np.isnan(xc)
        ## if not nm.any():
        ##     y[sl] = xc
        ##     continue
        ## a = np.arange(xc.shape[1])
        ## for row_x, row_y, nan_row in zip(xc, yc, nm):
        ##     fv = np.nanmean(row_x)
        ##     f = interp1d(a[~nan_row], row_x[~nan_row], kind=kind,
        ##                  bounds_error=False, fill_value=fv)
        ##     row_y[~nan_row] = row_x[~nan_row]
        ##     row_y[nan_row] = f( a[nan_row] )
        ## y[sl] = yc
