#!/usr/bin/env python
from fast_scroller.loading import VisLauncher

if __name__ == '__main__':
    v = VisLauncher(headstage='mux7', chan_map='ratv4_mux6')
    v.configure_traits()  