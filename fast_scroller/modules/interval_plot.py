import numpy as np
from traits.api import Button, Bool
from traitsui.api import View, Group, UItem, Label

import ecogana.anacode.seaborn_lite as sns

from .base import PlotsInterval

class IntervalTraces(PlotsInterval):
    name = 'Plot Window'
    plot = Button('Plot')
    new_figure = Bool(True)
    label_channels = Bool(False)

    def _plot_fired(self):
        x, y = self.parent._qtwindow.current_data()
        x = x[0]
        dy = self.parent.y_spacing
        f, ax = self._get_fig()
        clr = self._colors[self._axplots[ax]]
        y_levels = np.arange(len(y)) * dy
        ax.plot(x, y.T + y_levels, lw=0.5, color=clr)
        sns.despine(ax=ax)
        if not self.label_channels:
            ax.set_ylabel('Amplitude (mV)')
        else:
            ax.set_yticks( y_levels )
            ii, jj = self.parent._qtwindow.chan_map.to_mat()
            labels = map(str, zip(ii, jj))
            ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Time (s)')
        f.tight_layout()
        try:
            f.canvas.draw_idle()
        except:
            pass

    def default_traits_view(self):
        v = View(
            Group(
                UItem('plot'),
                Label('Label Channels'),
                UItem('label_channels'),
                label='Plot current window'
                )
            )
        return v
