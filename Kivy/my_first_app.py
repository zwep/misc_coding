# fucking Kivy yeah

from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

import os
import sys

from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.button import Label

from kivy.uix.widget import Widget

from kivy.uix.floatlayout import FloatLayout

from kivy.uix.gridlayout import GridLayout

from kivy.uix.boxlayout import BoxLayout

#StackLayout
# stacking widgets...
#PageLayout
# here you can flip over the 'pages'


loc_code = r'C:\Users\C35612.LAUNCHER\IdeaProjects\Experiments\Kivy'
loc_lib = r'C:\Users\C35612.LAUNCHER\IdeaProjects\NLP'

#os.chdir(loc_lib)
sys.path.append(loc_lib)
from helper.frequencysummarizer import *

os.chdir(loc_code)

""" 
TestApp is the first example of an app...
"""

class TestApp(App):
    def build(self):
        return Label()		
#TestApp().run()

""" 
CustomWidgetApp shows that the Custom versions 
(defined in a separate .kv file) are to be prefered over defeind ones 
"""

class CustomWidget(Widget):
	pass
	
class CustomWidgetApp(App):
	def build(self):
		return CustomWidget()
		
customWidget = CustomWidgetApp()
#customWidget.run()

""" 
Floating stuff/// that shape according to the size of the window
"""

class FloatingApp(App):
	def build(self):
		return FloatLayout()
		
fl = FloatingApp()
# fl.run()

""" 
Grid stuff
"""

class Henkiseensteen(GridLayout):
	pass

class GridLayoutaApp(App):
	def build(self):
		return Henkiseensteen()
		
gl = GridLayoutaApp()
# gl.run()

""" 
Box stuff
"""

class BoxLayoutApp(App):
	def build(self):
		return BoxLayout()
		
bl = BoxLayoutApp()
# bl.run()

""" 
Calculator stuff
"""

class CalcGridLayout(GridLayout):
 
    # Function called when equals is pressed
    def calculate(self, calculation):
        if calculation:
            try:
                # Solve formula and display it in entry
                # which is pointed at by display
                self.display.text = str(eval(calculation))
            except Exception:
                self.display.text = "Error"
 
class CalculatorApp(App):
 
    def build(self):
        return CalcGridLayout()
 
calcApp = CalculatorApp()
calcApp.run()


""" 
Summarizer stuff
"""

class CallGridSummarizer(GridLayout):

	# Function called when equals is pressed
	def summarize(self, field_of_text):
		fs = FrequencySummarizer()
		if field_of_text:
			try:				 
				self.display.text = '\n '.join(fs.summarize(field_of_text,2))
			except Exception:
				self.display.text = "Error"
	 
class SummarizerApp(App):
    def build(self):
        return CallGridSummarizer()
 
sumApp = SummarizerApp()
# sumApp.run()
	