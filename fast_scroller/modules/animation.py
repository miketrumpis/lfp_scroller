import sys
import time

from pyqtgraph.Qt import QtCore

from traits.api import Button, Enum
from traitsui.api import View, VGroup, Item
from pyface.timer.api import Timer

from .base import VisModule

__all__ = ['AnimateInterval']

class AnimateInterval(VisModule):
    name = 'Animate window'
    anim_frame = Button('Animate')
    anim_time_scale = Enum(50, [0.5, 1, 5, 10, 20, 50, 100])

    def __step_frame(self):
        n = self.__n
        x = self.__x
        y = self.__y
        if n >= self.__n_frames:
            self._atimer.Stop()
            return
        t0 = time.time()
        scaled_dt = self.anim_time_scale * (x[1] - x[0])
        try:
            self.parent._qtwindow.set_image_frame(x=x[n], frame_vec=y[:,n])
        except IndexError:
            self._atimer.Stop()
        QtCore.QCoreApplication.instance().processEvents()
        # calculate the difference between the desired interval
        # and the time it just took to draw (per "frame")
        elapsed = time.time() - t0
        t_pause = scaled_dt - elapsed / self.__f_skip
        if t_pause < 0:
            self.__f_skip += 1
            sys.stdout.flush()
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
        if hasattr(self, '_atimer') and self._atimer.IsRunning():
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

    def default_traits_view(self):
        v = View(
            VGroup(
                Item('anim_time_scale', label='Divide real time'),
                Item('anim_frame'),
                label='Animate Frames'
                )
            )
        return v
