import sounddevice as sd
import numpy as np
from multiprocessing import Process
from traits.api import Button, Enum, Bool, Float
from traitsui.api import View, HGroup, VGroup, UItem, Item, Label, EnumEditor, RangeEditor
from .base import VisModule

__all__ = ['SonifySeries']


class SonifySeries(VisModule):
    name = 'Sonify waveforms'
    play = Button('Play sound')
    speed_up = Enum(1, (0.25, 0.5, 1, 2, 4))
    average = Bool(False)
    volume = Float(1)

    def _play_fired(self):
        self.playback()

    def playback(self):
        x, y = self.curve_manager.interactive_curve.current_data(visible=True, full_xdata=True)
        if self.average:
            sound_wave = y.mean(axis=0)
            print(f'averaged {len(y)} channels for playback')
        else:
            chan_map = self.curve_manager.interactive_curve.map_curves(self.chan_map)
            i, j = list(map(float, self.channel.split(',')))
            sound_wave = y[chan_map.lookup(i, j)]
            print(f'playback for channel {i}, {j}')
        fs = 1 / self.curve_manager.interactive_curve.dx
        sound_wave = sound_wave - sound_wave.min()
        sound_wave = sound_wave / sound_wave.max() * self.volume

        sd.play(sound_wave, samplerate=fs * self.speed_up)
        # proc = Process(target=sd.play, args=(sound_wave,), kwargs=dict(samplerate=fs * self.speed_up))
        # proc.start()


    def default_traits_view(self):
        v = View(
            HGroup(
                VGroup(
                    Label('Channel to plot'),
                    UItem('channel')
                ),
                VGroup(
                    Item('speed_up', label='Playback rate'),
                    Item('average', label='Average channels')
                ),
                VGroup(
                    Item('volume', label='Volume', editor=RangeEditor(low=0, high=1)),
                    UItem('play')
                )
            )
        )
        return v
