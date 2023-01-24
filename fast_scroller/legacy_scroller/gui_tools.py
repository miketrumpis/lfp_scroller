import numpy as np
from traits.api import Float
import matplotlib.cm as cm
from . import plot_modules as pm


class EvokedPlot(pm.StaticFunctionPlot):
    """
    This is an instance of a StaticFunctionPlot of plot_modules, meaning
    that it has many Traits-ified attributes (e.g. xlim and ylim have
    listeners to automatically redraw the view). It plots multiple
    timeseries and has GUI interaction.

    In this case, the GUI interaction defines an "interval of interest"
    in the timeseries. Analyses of evoked potentials can take advantage
    of this user-defined interval.
    """

    peak_min = Float(0.005)
    peak_max = Float(0.025)

    def create_fn_image(self, x, t=None, **plot_line_props):
        # operates in the following modes:
        # * plots all one color (1D or 2D arrays)
        #  - color argument is single item
        # * plot in a gradient of colors (for 2D arrays)
        #  - color argument is an array or a colormap
        # * plot groups of timeseries by color (for 3D arrays)
        #  - color argument should be an array or a colormap
        #  - timeseries are grouped in the last axis
        if t is None:
            t = np.arange(len(x))

        if x.ndim == 1:
            x = np.reshape(x, (len(x), 1))

        ndim = x.ndim
        color = plot_line_props.pop('color', 'k')
        if isinstance(color, cm.colors.Colormap):
            plot_colors = color(np.linspace(0, 1, x.shape[-1]))
        elif isinstance(color, (np.ndarray, list, tuple)):
            plot_colors = color
        elif isinstance(color, str):
            plot_colors = (color,) * x.shape[-1]
        lines = list()
        if ndim == 2:
            self.ax.set_prop_cycle(color=list(plot_colors))
            lines = self.ax.plot(t, x, **plot_line_props)
        if ndim == 3:
            for group, color in zip(x.transpose(2, 0, 1), plot_colors):
                ln = self.ax.plot(t, group, color=color, **plot_line_props)
                lines.extend(ln)

        self.ax.set_xlabel('time from stimulus')
        pos = self.ax.get_position()
        self.ax.set_position([pos.x0, 0.15, pos.x1 - pos.x0, pos.y1 - .15])
        return lines

    def move_bar(self, *time):
        pass

    def connect_live_interaction(self):
        connections = pm.AxesScrubber.gen_event_associations(
            self, 'peak_min', 'peak_max',
            scrub_x=True, sense_button=3
        )
        super(EvokedPlot, self).connect_live_interaction(
            extra_connections=connections
        )

