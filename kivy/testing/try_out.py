from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import NoTransition
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

# <editor-fold desc='Loading all the libs'>

# Import stuff to do some text processing
from bs4 import BeautifulSoup
import xlrd
import os
import sys
sys.path.append(r'C:\Users\C35612.LAUNCHER\IdeaProjects\NLP')
from helper.processtext import *

import os
import sys
import re

# Import all kind of Kivy stuff...
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.button import Label
from kivy.uix.widget import Widget

from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.factory import Factory

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

from kivy.properties import StringProperty

import gensim

# </editor-fold>


class RootWidget(TabbedPanel):

    manager = ObjectProperty(None)
    text_input = ObjectProperty(None)
    text_loaded = False
    stored_summary = ''
    stored_keywords = ''
    stored_ner = ''
    stored_ngram = ''

    def switch_to(self, header):
        # set the Screen manager to load  the appropriate screen
        # linked to the tab head instead of loading content
        self.manager.current = header.screen
        # we have to replace the functionality of the original switch_to
        self.current_tab.state = "normal"
        header.state = 'down'
        self._current_tab = header


    def load(self, path, filename):
        """
        Loading files...

        :param path:
        :param filename:
        :return:
        """
        print(path)
        print(list(filename))
        with open(os.path.join(path, filename[0]), encoding='utf-8') as stream:
            text_BS = BeautifulSoup(stream, 'lxml')

        self.text_input.text = 'Succesfully loaded ' + filename[0]


class testkivyApp(App):

    def build(self):
        return RootWidget()


if __name__ == '__main__':
    testkivyApp().run()