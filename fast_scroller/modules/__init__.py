from traits.api import Enum, on_trait_change
from traitsui.api import View, Group, UItem

from .base import VisModule
from .animation import AnimateInterval
from .interval_plot import IntervalTraces
from .sorting_viewer import SortingViewer
from .spatial_field import SpatialVariance, ArrayVarianceTool
from .spatial_extra import ExtraSpatialVariance
from .spectra import IntervalSpectrum
from .spectrogram import IntervalSpectrogram
from .time_stamps import Timestamps
from .band_power_map import BandPowerMap
from .navigators import NavigatorManager
from .frame_filters import FrameFilters
from .series_filters import SeriesFiltering
from .sonify import SonifySeries


__all__ = ['AnimateInterval',
           'IntervalTraces',
           'SpatialVariance',
           'ExtraSpatialVariance',
           'ArrayVarianceTool',
           'IntervalSpectrum',
           'IntervalSpectrogram',
           'Timestamps',
           'SortingViewer',
           'BandPowerMap',
           'NavigatorManager',
           'FrameFilters',
           'SeriesFiltering',
           'SonifySeries',
           'YRange',
           'modules_by_name',
           'default_modules']


# This is a bit orphaned -- probably will disappear later
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


modules_by_name = {
    'Stacked traces plot': IntervalTraces,
    'Spatial variance': SpatialVariance,
    'Spatial variance (dev)': ExtraSpatialVariance,
    'Power spectrum': IntervalSpectrum,
    'Spectrogram': IntervalSpectrogram,
    'Animator': AnimateInterval,
    'Time stamps': Timestamps,
    'Band power map': BandPowerMap,
    'Navigators': NavigatorManager,
    'Spatial filters': FrameFilters,
    'Series filters': SeriesFiltering,
    'Sorting rasters': SortingViewer,
    'Sonify waveforms': SonifySeries,
    }

default_modules = ('Stacked traces plot',
                   'Power spectrum',
                   'Navigators',
                   'Animator',
                   'Spatial variance',
                   'Spectrogram')
