#!/usr/bin/env python
import matplotlib
matplotlib.use('Qt4Agg')
from fast_scroller.loading import VisLauncher

if __name__ == '__main__':
    v = VisLauncher(headstage='mux7', chan_map='ratv4_mux6_15row')
    v.configure_traits()
