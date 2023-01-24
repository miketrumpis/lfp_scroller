import os
import numpy as np
import matplotlib
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch, BoxStyle
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Str, Directory, Button, Instance, Float, Bool, Int, File, Enum
from traits.traits import Property
from traitsui.editors import FileEditor, RangeEditor, CustomEditor
from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.api import Group, HGroup, VGroup, UItem, Item, View, VSplit, Handler
from ecoglib.vis.colormaps import diverging_cm
from ecoglib.vis.ani import write_anim
from ecogdata.util import mkdir_p
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap, ChannelMapError
from fast_scroller.helpers import validate_file_path, Error

use = matplotlib.get_backend()

# Only really tested for QT4Agg backend
if use.lower() == 'qt5agg':
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
else:  # elif use.lower() == 'agg':
    # make this the fallback case
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.backend_bases import NavigationToolbar2 as NavigationToolbar

try:
    from traitsui.qt4.editor import Editor
except (ImportError, RuntimeError) as e:
    class Editor(object):
        pass


##############################################################################
########## Matplotlib to Traits Panel Integration ############################

class MiniNavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save', 'Subplots')]


def assign_canvas(editor):

    if isinstance(editor.object, Figure):
        mpl_fig = editor.object
    else:
        mpl_fig = editor.object.fig
    if hasattr(mpl_fig, 'canvas') and isinstance(mpl_fig.canvas, FigureCanvas):
        # strip this canvas, and close the originating figure?
        # num = mpl_fig.number
        # Gcf.destroy(num)
        return mpl_fig.canvas
    mpl_canvas = FigureCanvas(mpl_fig)
    return mpl_canvas


def _embedded_qt_figure(parent, editor, toolbar=True):
    try:
        from qtpy.QtWidgets import QVBoxLayout, QWidget
    except ImportError:
        from qtpy.QtGui import QVBoxLayout, QWidget

    panel = QWidget(parent.parentWidget())
    canvas = assign_canvas(editor)
    vbox = QVBoxLayout(panel)
    vbox.addWidget(canvas)
    if toolbar:
        toolbar = MiniNavigationToolbar(canvas, panel)
        vbox.addWidget(toolbar)
    panel.setLayout(vbox)
    return panel


embedded_figure = _embedded_qt_figure


class _MPLFigureEditor(Editor):
    """
    This class locates or provides a QT canvas to all MPL figures when drawn
    under the TraitsUI framework. This also works for a HasTraits object with a
    single MPL figure as .fig attribute.
    """

    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        if isinstance(self.value, Figure):
            mpl_fig = self.value
        else:
            mpl_fig = self.value.fig
        # if canvas is good, return it. If it's a dummy canvas, then re-set it
        if hasattr(mpl_fig, 'canvas') and isinstance(mpl_fig.canvas, FigureCanvas):
            return mpl_fig.canvas
        else:
            mpl_canvas = FigureCanvas(mpl_fig)
            return mpl_canvas
        # for Qt?
        # mpl_canvas.setParent(parent)


class MPLFigureEditor(BasicEditorFactory):

    klass = _MPLFigureEditor


class PingPongStartup(Handler):
    """
    This object can act as a View Handler for an HasTraits instance
    that creates Matplotlib elements after the GUI canvas is created.
    This handler simply calls the _post_canvas_hook() method on the
    HasTraits instance, which applies any finishing touches to the MPL
    elements.
    """

    def init(self, info):
        info.object._post_canvas_hook()


class SavesFigure(HasTraits):
    sfile = Str
    spath = Directory(os.path.abspath(os.curdir), exists=True)
    path_button = Button('fig dir')
    update = Button('Refresh')

    fig = Instance(Figure)
    save = Button('Save plot')
    dpi = Enum(400, (100, 200, 300, 400, 500, 600))

    _extensions = ('pdf', 'svg', 'eps', 'png')
    format = Enum('pdf', _extensions)

    # internal axes management
    # has_images = Bool(False)
    has_images = Property
    c_lo = Float
    c_hi = Float
    cmap_name = Str
    # has_graphs = Bool(False)
    has_graphs = Property
    y_lo = Float
    y_hi = Float
    tight_layout = Bool(False)

    @classmethod
    def live_fig(cls, fig, **traits):
        sfig = cls(fig, **traits)
        v = sfig.default_traits_view()
        v.kind = 'live'
        # sfig.edit_traits(view=v)
        sfig.configure_traits(view=v)
        return sfig

    def __init__(self, fig, **traits):
        super(SavesFigure, self).__init__(**traits)
        self.fig = fig
        if self.has_images:
            self._search_image_props()

        self.on_trait_change(self._clip, 'c_hi')
        self.on_trait_change(self._clip, 'c_lo')
        self.on_trait_change(self._cmap, 'cmap_name')
        if self.has_graphs:
            self._search_graph_props()
        self.on_trait_change(self._ylims, 'y_hi')
        self.on_trait_change(self._ylims, 'y_lo')

    def _search_image_props(self):
        if not self.has_images:
            return
        clim = None
        for ax in self.fig.axes:
            if ax.images:
                clim = ax.images[0].get_clim()
                cm = ax.images[0].get_cmap().name
                break
        if clim:
            self.trait_set(c_lo=clim[0], c_hi=clim[1], cmap_name=cm)

    def _search_graph_props(self):
        ylim = ()
        for ax in self.fig.axes:
            # is this specific enough?
            if not ax.images and ax.lines:
                ylim = ax.get_ylim()
                break
            if not ax.images and ax.collections:
                ylim = ax.get_ylim()
                break
        if ylim:
            self.trait_set(y_lo=ylim[0], y_hi=ylim[1])

    def _get_has_images(self):
        if not hasattr(self.fig, 'axes'):
            return False
        for ax in self.fig.axes:
            if hasattr(ax, 'images') and len(ax.images) > 0:
                return True
        return False

    def _get_has_graphs(self):
        if not hasattr(self.fig, 'axes'):
            return False
        for ax in self.fig.axes:
            if hasattr(ax, 'lines') and len(ax.lines) > 0:
                return True
            if hasattr(ax, 'collections') and len(ax.collections) > 0:
                return True
        return False

    def _ylims(self):
        for ax in self.fig.axes:
            if not ax.images and ax.lines:
                ax.set_ylim(self.y_lo, self.y_hi)
        self.fig.canvas.draw()

    def _clip(self):
        for ax in self.fig.axes:
            for im in ax.images:
                im.set_clim(self.c_lo, self.c_hi)
        self.fig.canvas.draw()

    def _cmap(self):
        name = self.cmap_name
        # Check for a "_z<p>" pattern, which signals a z score map with compressed saturation
        name_parts = name.split('_')
        z_map = False
        if len(name_parts) > 1:
            z_part = name_parts[-1]
            if len(z_part) and z_part[0] == 'z':
                # take the _z code out of the name
                name = '_'.join(name_parts[:-1])
                z_map = True
                try:
                    p = float(z_part[1:])
                except ValueError:
                    p = 1
        # Go through the logic of looking for a valid colormap
        try:
            colors = cm.get_cmap(name)
        except ValueError:
            # try to evaluate the string as a function in colormaps module
            try:
                code = 'plotters.sns.' + name
                colors = eval(code)
            except:
                return
        if z_map:
            colors = diverging_cm(self.c_lo, self.c_hi, cmap=colors, compression=p)
        for ax in self.fig.axes:
            for im in ax.images:
                im.set_cmap(colors)
        self.fig.canvas.draw()

    def _save_fired(self):
        if not (self.spath or self.sfile):
            return
        if self.spath[-1] != '/':
            self.spath = self.spath + '/'
        pth = os.path.dirname(self.spath)
        mkdir_p(pth)

        f, e = os.path.splitext(self.sfile)
        if e in self._extensions:
            self.sfile = f

        ext_sfile = self.sfile + '.' + self.format
        self.fig.savefig(os.path.join(self.spath, ext_sfile), dpi=self.dpi)

    def _update_fired(self):
        self._search_image_props()
        self._search_graph_props()

    def _post_canvas_hook(self):
        if self.tight_layout:
            self.fig.tight_layout()

    def default_traits_view(self):
        # The figure is put in a panel with correct fig-width and fig-height.
        # Using negative numbers locks in the size. It appears that using
        # positive numbers enforces a minimum size.
        fig = self.fig
        fh = int(fig.get_figheight() * fig.get_dpi())
        fw = int(fig.get_figwidth() * fig.get_dpi())
        traits_view = View(
            VSplit(
                UItem(
                    'fig', editor=CustomEditor(embedded_figure),
                    resizable=True, height=fh, width=fw
                ),
                Group(
                    HGroup(
                        Item('spath', label='Figure path'),
                        # UItem('path_button'),
                        Item('sfile', style='simple', label='Image File')
                    ),
                    HGroup(
                        HGroup(
                            Item('dpi', label='DPI'),
                            Item('format'),
                            UItem('save'),
                            label='Image format'
                        ),
                        HGroup(
                            UItem('update'),
                            label='Refresh properties'
                        )
                    ),

                    HGroup(
                        VGroup(
                            HGroup(
                                Item('c_lo', label='Clip lo',
                                     enabled_when='has_images', width=4),
                                Item('c_hi', label='Clip hi',
                                     enabled_when='has_images', width=4),
                                enabled_when='has_images'
                            ),
                            Item('cmap_name', label='Colormap',
                                 enabled_when='has_images'),
                            enabled_when='has_images',
                            label='Control image properties'
                        ),
                        VGroup(
                            Item('y_lo', label='y-ax lo', width=4),
                            Item('y_hi', label='y-ax hi', width=4),
                            enabled_when='has_graphs',
                            label='Control axis properties'
                        )
                    )
                )
            ),
            resizable=True,
            handler=PingPongStartup
        )
        return traits_view


class MultiframeSavesFigure(SavesFigure):
    """
    Specialization of SavesFigure that has a plotting element
    with multiple frames that can be scanned-through.
    """

    _mx = Int(10)
    _mn = Int(0)
    mode = Int(0)  # Range(low='_mn', high='_mx')
    mode_name = 'Mode'
    mode_value = Property(Float, depends_on='mode')
    _has_ffmpeg = Bool
    video_file = File
    video_fps = Int(10)
    make_video = Button('Make video')

    def __init__(self, fig, frames, frame_index=(), **traits):
        import matplotlib.animation as anim
        traits['_has_ffmpeg'] = 'ffmpeg' in anim.writers.list()
        super(MultiframeSavesFigure, self).__init__(fig, **traits)
        self.frames = frames
        self._mx = len(frames)-1
        if not len(frame_index):
            frame_index = list(range(len(frames)))
        self.frame_index = frame_index

    def _get_mode_value(self):
        return np.round(self.frame_index[self.mode], decimals=2)

    @on_trait_change('mode')
    def change_frame(self):
        # has to assume fig.axes[0].images[0] has an image!
        im = self.fig.axes[0].images[0]
        im.set_array(self.frames[self.mode])
        self.fig.canvas.draw_idle()

    def _make_video_fired(self):
        if not validate_file_path(self.video_file):
            ev = Error(
                error_msg='Invalid video file:\n{0}'.format(self.video_file)
            )
            ev.edit_traits()
            return
        mode = self.mode

        def step_fn(n):
            self.mode = n
            return (self.fig.axes[0].images[0],)
        write_anim(
            self.video_file, self.fig, step_fn, self._mx,
            quicktime=True
        )
        self.mode = mode

    def default_traits_view(self):
        v = super(MultiframeSavesFigure, self).default_traits_view()
        vsplit = v.content.content[0]

        if self._has_ffmpeg:
            vpanel = HGroup(
                Item('video_file', editor=FileEditor(dialog_style='save')),
                Item('video_fps', label='FPS',
                     editor=RangeEditor(low=1, high=100, mode='spinner')),
                UItem('make_video')
            )
            vsplit.content.insert(1, vpanel)
        panel = HGroup(
            Item('mode', label='Scroll frames',
                 editor=RangeEditor(low=self._mn, high=self._mx)),
            UItem('mode', visible_when='_mx > 101',
                  editor=RangeEditor(
                      low=self._mn, high=self._mx, mode='slider'
                  )),
            Item('mode_value', style='readonly',
                 label='Current frame: {0}'.format(self.mode_name))
        )
        vsplit.content.insert(1, panel)
        return v

    @staticmethod
    def named_toggle(name):
        return MultiframeSavesFigure(mode_name=name)


class ArrayMap(HasTraits):
    """
    A simple wrapper of an MPL figure. Has a .fig and works
    with MPLFigureEditor

    The figure itself is a sensor vector embedded in the geometry
    of the electrode array. Sites can be clicked and selected to
    modify the "selected_site" Trait
    """
    selected_site = Int(-1)

    def __init__(self, chan_map, labels=None, vec=None, ax=None, map_units=None, cbar=True,
                 mark_site=True, **plot_kwargs):
        # the simplest instantiation is with a vector to plot
        self.labels = labels
        self._clim = plot_kwargs.pop('clim', None)
        chan_image = chan_map.image(vec, cbar=cbar, ax=ax, clim=self._clim, **plot_kwargs)
        if cbar:
            self.fig, self.cbar = chan_image
        else:
            self.fig = chan_image
            self.cbar = None
        self.ax = self.fig.axes[0]
        self.ax.axis('image')
        if self.cbar:
            if labels is not None:
                self.cbar.set_ticks(np.arange(0, len(labels)))
                self.cbar.set_ticklabels(labels)
            if map_units is not None:
                self.cbar.set_label(map_units)

        super(ArrayMap, self).__init__()

        self._coord_map = isinstance(chan_map, CoordinateChannelMap)
        # both these kinds of maps update in the same way :D
        if self._coord_map:
            self._map = self.ax.collections[-1]
        else:
            self._map = self.ax.images[-1]
        self.chan_map = chan_map
        self._box = None
        self._mark_site = mark_site

        # if (ax is None):
        #     if vec is None:
        #         vec = np.ones(len(chan_map))
        #     cmap = traits_n_kws.pop('cmap', cm.Blues)
        #     origin = traits_n_kws.pop('origin', 'upper')
        #     fsize = np.array(chan_map.geometry[::-1], 'd') / 3.0
        #     self.fig = Figure(figsize=tuple(fsize))
        #     self.ax = self.fig.add_subplot(111)
        #     self._map = self.ax.imshow(
        #         chan_map.embed(vec), cmap=cmap, origin=origin,
        #         clim=self._clim
        #     )
        #     self.ax.axis('image')
        #     self.cbar = self.fig.colorbar(
        #         self._map, ax=self.ax, use_gridspec=True
        #     )
        #     if labels is not None:
        #         self.cbar.set_ticks(np.arange(0, len(labels)))
        #         self.cbar.set_ticklabels(labels)
        #     elif map_units is not None:
        #         self.cbar.set_label(map_units)
        # elif ax:
        #     self.ax = ax
        #     self.fig = ax.figure


    def click_listen(self, ev):
        try:
            i, j = ev.ydata, ev.xdata
            if i is None or j is None:
                raise TypeError
            if not self._coord_map:
                i, j = list(map(round, (i, j)))
        except TypeError:
            if ev.inaxes is None:
                self.selected_site = -1
            return
        try:
            self.selected_site = self.chan_map.lookup(i, j)
        except ChannelMapError:
            self.selected_site = -1

    @on_trait_change('selected_site')
    def _move_box(self):
        if not self._mark_site:
            return
        try:
            # negative index codes for outside array
            if self.selected_site < 0:
                raise IndexError
            i, j = self.chan_map.rlookup(self.selected_site)
            if self._box:
                self._box.remove()
            box_size = self.chan_map.site_combinations.dist.min() if self._coord_map else 1
            style = BoxStyle('Round', pad=0.3 * box_size, rounding_size=None)
            self._box = FancyBboxPatch(
                (j - box_size / 2.0, i - box_size / 2.0), box_size, box_size, boxstyle=style,
                fc='none', ec='k', transform=self.ax.transData,
                clip_on=False
            )
            self.ax.add_patch(self._box)
        except IndexError:
            if self._box:
                self._box.remove()
                self._box = None
            pass
        finally:
            self.fig.canvas.draw()

    def update_map(self, scores, c_label=None, **extra):
        "Update map image given new set of scalars from the sensor vector"
        if 'clim' not in extra:
            extra['clim'] = self._clim
        if not self._coord_map:
            if scores.shape != self.chan_map.geometry:
                scores = self.chan_map.embed(scores)
        elif len(scores) != len(self.chan_map):
            raise ValueError("Can't plot vector length {} to a coordinate map.".format(len(scores)))
        self._map.set_array(scores)
        self._map.update(extra)
        if self.cbar:
            if c_label is not None:
                self.cbar.set_label(c_label)
        try:
            if self.cbar:
                self.cbar.draw_all()
            self.fig.canvas.draw()
        except:
            # no canvas? no problem
            pass

