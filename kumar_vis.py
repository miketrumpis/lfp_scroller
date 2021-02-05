#!/usr/bin/env python

import os
import numpy as np
from fast_scroller.h5scroller import FastScroller
from fast_scroller.new_scroller import VisWrapper
from fast_scroller.modules import SeriesFiltering, IntervalSpectrum, IntervalSpectrogram, IntervalTraces, Timestamps
from ecogdata.devices.electrode_pinouts import get_electrode_map
from rhdlib import RHDFile
from ecogdata.parallel.mproc import make_stderr_logger


try:
    get_ipython()
    using_ipython = True
except NameError:
    using_ipython = False

# rhd_file = '/Users/mike/experiment_data/Human_uECoG/S15_2020-08-13_IntraOp/S15_intraop_200813_103521.rhd'
rhd_file = 'S13_OR_191010_184237.rhd'
rhd = RHDFile(rhd_file)
print('Loading arrays...', end=' ', flush=True)
arrays = rhd.to_arrays()
print('done')
channel_map, grounds, refs = get_electrode_map('psv_244_rhd')
rhd_volt_scale = 0.195 * 1e-6
amp_data = arrays['amplifier_data']
if 'board_adc_data' in arrays:
    adc_data = arrays['board_adc_data']
else:
    adc_data = None
signal_channels = np.setdiff1d(np.arange(len(amp_data)), grounds)
samp_rate = rhd.header['frequency_parameters']['amplifier_sample_rate']

# y-units are Volts, x-units are seconds
y_gap = 50e-6  # 50 uV spacing
# Use "info" for a lot of info, else "error"
# log_level = 'error'
log_level = 'info'
with make_stderr_logger(log_level):

    qt_scroller = FastScroller(amp_data, rhd_volt_scale, 50e-6, channel_map, amp_data.mean(0),
                              x_scale=1 / samp_rate, load_channels=signal_channels)


    modules = [Timestamps, SeriesFiltering, IntervalSpectrum, IntervalTraces, IntervalSpectrogram]
    vis = VisWrapper(qt_scroller, modules=modules, x_scale=1 / samp_rate, chan_map=channel_map, y_spacing=y_gap * 1e6,
                     recording=os.path.split(rhd_file)[1])
    if using_ipython:
        vis.edit_traits()
    else:
        vis.configure_traits()
