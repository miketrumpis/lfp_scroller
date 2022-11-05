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

__all__ = ['AnimateInterval']


class AnimateInterval(VisModule):
    name = 'Animate window'
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.1, 0.25, 0.5, 1, 2, 4, 10, 20, 50, 100, 200, 500])
    _has_ffmpeg = Bool(False)
    frame_skip = Int(1)
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
            self._atimer.stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self.parent._qtwindow.set_image_frame(x=x[n], frame_vec=y[:, n])
        except IndexError:
            self._atimer.stop()
        QtCore.QCoreApplication.instance().processEvents()
        # calculate the difference between the desired interval
        # and the time it just took to draw (per "frame")
        elapsed = time.time() - t0
        t_pause = scaled_dt - elapsed / self.__f_skip
        # If we're behind schedule, then increase the frame skip size
        # AND calculate how many frames we're currently behind
        if t_pause < 0:
            self.__f_skip *= 2
            extra_frames = int(-t_pause / scaled_dt) + 10
        else:
            # timer is in the middle of API change
            self._atimer.interval = t_pause
            # check to see if the frame skip can be decreased (e.g.
            # real-time is slowed down)
            while elapsed / max(1, self.__f_skip - 1) < scaled_dt:
                self.__f_skip = int(max(1, self.__f_skip * 0.8))
                if self.__f_skip == 1:
                    break
        self.frame_skip = int(self.__f_skip)
        self.__n += self.__f_skip + extra_frames

    def _anim_frame_fired(self):
        if hasattr(self, '_atimer'):
            if self._atimer.active:
                self._atimer.stop()
                return

        x, self.__y = self.curve_manager.interactive_curve.current_data(full_xdata=False)
        self.__x = x
        dt = self.__x[1] - self.__x[0]
        # aim for ~100 fps
        raw_rate = 1 / dt
        skips = int(raw_rate / 100)
        # 10 ms = 100 fps
        timer_rate = 10
        self.__f_skip = skips
        self.__n_frames = self.__y.shape[1]
        self.__n = 0
        self._atimer = Timer(timer_rate, self.__step_frame)

    def _get_clim(self, array):
        if self.clim == 'full':
            return (array.min(), array.max())
        if self.clim.endswith('%'):
            clim = self.clim.replace('[', '').replace(']', '').replace('%', '')
            p_lo, p_hi = map(float, clim.split('-'))
            print(p_lo, p_hi)
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

        x, y = self.curve_manager.interactive_curve.current_data(full_xdata=False)
        y *= 1e6
        dt = x[1] - x[0]
        # fps is sampling frequency divided by time scale dilation
        fps = (dt * self.anim_time_scale) ** -1.0
        chan_map = self.chan_map
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
                    Item('frame_skip', label='Skip frames', style='readonly'),
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
