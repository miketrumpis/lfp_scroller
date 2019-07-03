import os
import time
from multiprocessing import Process
import numpy as np
from pyqtgraph.Qt import QtCore

from traits.api import Button, Enum, Bool, Int, File
from traitsui.api import View, VGroup, HGroup, UItem, \
     Item, FileEditor, RangeEditor
from pyface.timer.api import Timer

import ecoglib.vis.ani as ani

from .base import VisModule, colormaps
from ..helpers import Error, validate_file_path
from .. import pyf_new_api

__all__ = ['AnimateInterval']

class AnimateInterval(VisModule):
    name = 'Animate window'
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.1, 0.5, 1, 5, 10, 20, 50, 100, 200, 500])
    _has_ffmpeg = Bool(False)
    write_frames = Button('Write movie')
    drop_video_frames = Int(1)
    video_file = File(
        os.path.join(os.path.abspath(os.curdir), 'vid.mp4')
        )
    cmap = Enum('gray', colormaps)
    clim = Enum('display', ('display', '[2-98]%', '[1-99]%', 'full'))

    def __init__(self, **traits):
        import matplotlib.animation as anim
        traits['_has_ffmpeg'] = 'ffmpeg' in anim.writers.list()
        super(AnimateInterval, self).__init__(**traits)

    def __step_frame(self):
        n = self.__n
        x = self.__x
        y = self.__y
        if n >= self.__n_frames:
            if pyf_new_api:
                self._atimer.stop()
            else:
                self._atimer.Stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self.parent._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            if pyf_new_api:
                self._atimer.stop()
            else:
                self._atimer.Stop()
        QtCore.QCoreApplication.instance().processEvents()
        # calculate the difference between the desired interval
        # and the time it just took to draw (per "frame")
        elapsed = time.time() - t0
        t_pause = scaled_dt - elapsed / self.__f_skip
        if t_pause < 0:
            self.__f_skip += 1
        else:
            # timer is in the middle of API change
            if pyf_new_api:
                self._atimer.interval = t_pause
            else:
                self._atimer.setInterval(t_pause * 1000.0)
            # check to see if the frame skip can be decreased (e.g.
            # real-time is slowed down)
            while elapsed / max(1, self.__f_skip - 1) < scaled_dt:
                self.__f_skip = max(1, self.__f_skip - 1)
                if self.__f_skip == 1:
                    break
        self.__n += self.__f_skip

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer'):
            if pyf_new_api and self._atimer.active:
                self._atimer.stop()
                return
            elif not pyf_new_api and self._atimer.IsRunning():
                self._atimer.Stop()
                return
        
        x, self.__y = self.parent._qtwindow.current_data()
        self.__f_skip = 1
        self.__x = x[0]
        dt = self.__x[1] - self.__x[0]
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        self._atimer = Timer( self.anim_time_scale * dt * 1000,
                              self.__step_frame )

    def _get_clim(self, array):
        if self.clim == 'full':
            return (array.min(), array.max())
        if self.clim.endswith('%'):
            clim = self.clim.replace('[', '').replace(']', '').replace('%', '')
            p_lo, p_hi = map(float, clim.split('-'))
            print p_lo, p_hi
            return np.percentile(array.ravel(), [p_lo, p_hi])
        else:
            clim = self.parent._qtwindow.cb.axis.range
            return clim[0] * 1e6, clim[1] * 1e6

    def _write_frames_fired(self):
        if not validate_file_path(self.video_file):
            ev = Error(
                error_msg='Invalid video file:\n{0}'.format(self.video_file)
                )
            ev.edit_traits()
            return

        x, y = self.parent._qtwindow.current_data()
        y *= 1e6
        x = x[0]
        dt = x[1] - x[0]
        # fps is sampling frequency divided by time scale dilation
        fps = (dt * self.anim_time_scale) ** -1.0
        chan_map = self.parent._qtwindow.chan_map
        if self.drop_video_frames > 1:
            x = x[::self.drop_video_frames]
            y = y[..., ::self.drop_video_frames]
            fps /= float(self.drop_video_frames)
        frames = chan_map.embed(y.T, axis=1)
        clim = self._get_clim(y)

        args = (frames, self.video_file)
        kwargs = dict(timer='s', time=x, fps=fps, title='Scroller video', quicktime=True, colorbar=True,
                      cbar_label='uV', cmap=self.cmap, clim=clim, origin='upper', qtdpi=100)
        proc = Process(target=ani.write_frames, args=args, kwargs=kwargs)
        proc.start()

    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(
                    Item('anim_time_scale', label='Divide real time'),
                    Item('anim_frame'),
                    label='Animate Frames'
                    ),
                HGroup(
                    VGroup(
                        Item('video_file', label='MP4 File',
                            editor=FileEditor(dialog_style='save')),
                        UItem('write_frames')
                        ),
                    VGroup(
                        Item('cmap', label='Colormap'),
                        Item('clim', label='Color limit mode'),
                        Item('drop_video_frames',
                             label='Frame drop rate',
                             editor=RangeEditor(low=1, high=100,
                                                mode='spinner')),
                        ),
                    visible_when='_has_ffmpeg'
                    )
                )
            )
        return v
