# -*- coding: utf-8 -*-
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, StringProperty, NumericProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
import os
from random import sample, randint
from string import ascii_lowercase


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    """ Adds selection and focus behaviour to the view."""


class SelectableLabel(RecycleDataViewBehavior, Label):
    """Add selection support to the Label """
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        """Catch and handle the view changes """
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        """ Add selection on touch down """
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """Respond to the selection of items in the view. """
        self.selected = is_selected
        if is_selected:
            print("selection changed to {0}".format(rv.data[index]))
        else:
            print("selection removed for {0}".format(rv.data[index]))

class Test(BoxLayout):
    """ Main Kivy class for creating the initial BoxLayout """

    various_dict = ['D:\\', 'D:\\temp', 'D:\\temp\\reg_doc']
    n = len(various_dict)

    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

        # Set media_list data
        self.ids.rv.data = [{'text': str(x)} for x in range(100)]
    #
    # def populate(self):
    #     derp = os.listdir(self.various_dict[randint(0, self.n - 1)])
    #     self.rv.data = [{'value': x} for x in derp]
    #     self.rv.data[0]['selected'] = 1
    #     print(self.rv.data[0])
    #
    # def sort(self):
    #     self.rv.data = sorted(self.rv.data, key=lambda x: x['value'])
    #
    # def clear(self):
    #     self.rv.data = []
    #
    # def insert(self, value):
    #     self.rv.data.insert(0, {'value': value or 'default value'})
    #     print(value)
    #
    # def update(self, value):
    #     if self.rv.data:
    #         self.rv.data[0]['value'] = value or 'default new value'
    #         self.rv.refresh_from_data()
    #
    # def remove(self):
    #     if self.rv.data:
    #         self.rv.data.pop(0)

class TestApp(App):
    def build(self):
        return Test()


if __name__ == '__main__':
    TestApp().run()