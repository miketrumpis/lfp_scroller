#!/usr/bin/env python

import sys
from fast_scroller.loading import VisLauncher
from ecogdata.parallel.mproc import make_stderr_logger

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1].endswith('v'):
        log_level = 'info'
    else:
        log_level = None
    v = VisLauncher(headstage='mux7', chan_map='ratv4_mux6_15row')
    with make_stderr_logger(log_level):
        v.configure_traits()
