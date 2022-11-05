import sounddevice as sd
import numpy as np
import os
from scipy.io.wavfile import write as write_wav
from typing import Tuple
from multiprocessing import Process
from threading import Thread
from traits.api import Button, Enum, Bool, Float, File
from traitsui.api import View, HGroup, VGroup, UItem, Item, Label, FileEditor, RangeEditor
from .base import VisModule
from . import AnimateInterval

__all__ = ['SonifySeries']


class SonifySeries(VisModule):
    name = 'Sonify waveforms'
    play = Button('Play sound')
    save = Button('Save sound')
    speed_up = Enum(1, (0.25, 0.5, 1, 2, 4))
    average = Bool(False)
    scroll_frames = Bool(True)
    wav_file = File(os.path.join(os.path.abspath(os.curdir), 'audio.wav'))
    volume = Float(1)

    def _play_fired(self):
        self.playback()

    def _save_fired(self):
        self.save_audio()

    def get_audio(self) -> Tuple[np.ndarray, float]:
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
        return sound_wave, fs

    def save_audio(self):
        sound_wave, fs = self.get_audio()
        write_wav(self.wav_file, int(np.round(fs)), sound_wave.astype('f'))

    def playback(self):
        sound_wave, fs = self.get_audio()
        # proc = Process(target=sd.play, args=(sound_wave,), kwargs=dict(samplerate=int(fs * self.speed_up)))
        proc = Thread(target=sd.play, args=(sound_wave,), kwargs=dict(samplerate=int(fs * self.speed_up)))
        if self.scroll_frames:
            # Make an Animator to sweep across the current frames (??) while sound plays
            animator = AnimateInterval(parent=self.parent,
                                       chan_map=self.chan_map,
                                       curve_manager=self.curve_manager,
                                       anim_time_scale=1.0 / self.speed_up)
            animator._anim_frame_fired()
        # sd.play(sound_wave, samplerate=int(np.round(fs * self.speed_up)))
        proc.start()

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
                ),
                Item('scroll_frames', label='Scroll through frames'),
                VGroup(
                    Item('wav_file', label='WAV File',
                         editor=FileEditor(dialog_style='save')),
                    Item('save', label='Save file')
                )
            )
        )
        return v
