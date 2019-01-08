from scipy.ndimage import gaussian_filter
from traits.api import Button, Str, Float, HasTraits, Instance, List, Enum
from traitsui.api import View, HGroup, Group, UItem, Handler, ListEditor

from .base import VisModule
from ..filtering import AutoPanel

__all__ = ['FrameFilters']

class FilterBuilder(AutoPanel):
    orientation = 'horizontal'


class GaussianFilter(FilterBuilder):
    sigma = Float(1.0)


    def make_filter(self):
        fn = lambda x: gaussian_filter(x, self.sigma)
        return fn


class FilterHandler(Handler):

    def object_name_changed(self, info):
        name = info.object.name.lower()
        if name == 'gaussian':
            info.object.menu = GaussianFilter()


class SpatialFilter(HasTraits):
    name = Str
    menu = Instance(FilterBuilder)

    view = View(
        HGroup(
            UItem('name', style='readonly'),
            Group(UItem('menu', style='custom'), label='Filter params')
        ),
        handler=FilterHandler
    )


filter_tabs = ListEditor(
    use_notebook=True,
    dock_style='tab',
    page_name='.name',
    selected='selected'
)


filter_types = ('Gaussian',)

class FrameFilters(VisModule):
    name = 'Frame Filters'
    filters = List(SpatialFilter)
    selected = Instance(SpatialFilter)
    filter_type = Enum(filter_types[0], filter_types)
    add_filter = Button('Add filter')
    set_filter = Button('Set filter')
    unset_filter = Button('Unset filter')


    def add_filter_by_type(self, filter_type):
        filt = SpatialFilter(name=filter_type)
        self.filters.append(filt)
        if len(self.filters) == 1:
            self.selected = filt


    def _add_filter_fired(self):
        self.add_filter_by_type(self.filter_type)


    def _set_filter_fired(self):
        fn = self.selected.menu.make_filter()
        self.parent._qtwindow.frame_filter = fn


    def _unset_filter_fired(self):
        self.parent._qtwindow.frame_filter = None


    def default_traits_view(self):
        v = View(
            HGroup(
                Group(
                    UItem('filter_type'),
                    UItem('add_filter')
                ),
                Group(
                    UItem('set_filter'),
                    UItem('unset_filter')
                ),
                Group(
                    UItem('filters', style='custom', editor=filter_tabs),
                    show_border=True
                ),
                springy=True
            ),
            title=self.name
        )
        return v