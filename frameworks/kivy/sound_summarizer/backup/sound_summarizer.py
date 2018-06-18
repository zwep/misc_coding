# fucking Kivy yeah
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

import os
import sys
import pyaudio
import wave
import matplotlib.pyplot as plt
import winsound
import re

import nltk
import nltk.corpus
import nltk
import nltk.tokenize


import glob 

import speech_recognition as sr

import numpy as np
import pandas as pd

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.button import Label
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.properties import ObjectProperty

from pocketsphinx import LiveSpeech

from kivy.core.window import Window
 
"""
Define locations
"""
 
loc_code = r'C:\Users\C35612.LAUNCHER\Testing_code\Kivy_app\sound_summarizer' 

sys.path.append(r'C:\Users\C35612.LAUNCHER\Testing_code\Kivy_app\sound_summarizer\lib')

from loadrss import *
from processtext import *
from TextAnalyzer import *
 

Window.clearcolor = (0, .44, .4, .2)

"""
 FUNCTIONS
"""

def transform_to_color(x):
	"""
	Based on the amount of unique points in the label list... 
	Return a list that contains a translation to distinct colors
	"""
	unique_x = np.unique(x)
	n = len(unique_x)
	color_x = list(iter(plt.cm.rainbow(np.linspace(0,1,n))))
	dict_color   = dict(zip(unique_x,color_x))
	x_to_color = list(map(dict_color.get,x))
	return x_to_color

def subset_list(y,n_min):	
	"""
	Subsets a list with easy list comprehension... But returns
	the subset list, as well as the index. 
	""" 
	y_index = []
	y_sub = []
	y_index_sub = [(i,x) for i,x in enumerate(y) if len(x) > n_min]
	if len(y_index_sub) != 0:
		y_index, y_sub = list(map(list,zip(*y_index_sub)))
	return y_index, y_sub


def wav_to_audio_data(file_name):	
		"""		
		Convert the recorded sound to an AudioData format
		This is needed for the recognition
		"""
		spf = wave.open(file_name,'r')
		
		audio_framerate = spf.getframerate()
		audio_smplwidth = spf.getsampwidth()

		#Extract Raw Audio from Wav File
		signal_bit = spf.readframes(-1)
		# Not returning singal_int for now.,,.
		spf_audio = sr.AudioData(signal_bit,sample_rate=audio_framerate,sample_width=audio_smplwidth)
		
		return spf_audio
		

os.chdir(loc_code)
 
class SoundSummarizer(GridLayout):
	"""
	Here we define a clas to record, pay, read and write data
	"""
	display = ObjectProperty(None)
	build_init = False
	
	def build(self,file_name=''):
		# initialize variables
		self.chunk = 1024
		self.format = pyaudio.paInt16
		self.channels = 1
		self.rate = 44100		
		if file_name:
			self.file_name = file_name + ".wav"
		else:
			self.file_name = re.sub('[:punc:]|\.','',str(pd.datetime.today())) + ".wav"
			
		self.news_article, self.news_article_label = get_url_content()
		# Deselect useless titles and stuff
		self.news_article = [x for x in self.news_article if len(x) > 80]
		self.random_article = 0
		self.build_init = True
		
		self.clean_text = self.news_article[self.random_article]
		
		# initialize classes
		self.port_audio = pyaudio.PyAudio()
		self.record_seconds = int()						
		self.sound_data = []
		self.audio_data = sr.AudioData(b'',1,1)
		self.recog = sr.Recognizer()
		
	def print_main_text(self,input):
		self.display[0].text = str(input)
		
	def print_text(self,input):
		self.display[1].text = str(input)
		
	def record_sound(self,record_seconds):
		"""
		With the inialized variables... records a sound
		"""
		if not self.build_init:
			self.build()
			
		stream = self.port_audio.open(format = self.format, 
								   channels = self.channels,
								   rate = self.rate,
								   input = True, 
								   frames_per_buffer = self.chunk)
		# WHy do we capture d = [] here while write data...
		# And what is the differennce between s.write() and wf.write()?
		#print("* recording")
		
		for i in range(0, (self.rate // self.chunk * record_seconds)): 
			data = stream.read(self.chunk)
			self.sound_data.append(data)			
		#print("* done")	
		#self.display[0].text = 'done'
		stream.stop_stream()
		stream.close()
		
		# convert the just recorded sound to AudioData
		self._sound_to_audio_data()
		self._write_sound()
		self.print_text('finished recording')
		
		# Get RSS feed
		# Show summarizer
	def _write_sound(self,file_name=''): 
		"""
		Writing the sound to a specified filename.
		When no filename is present, it uses the one that is initalized
		"""
		if file_name:
			wf = wave.open(file_name,'wb')
		else:
			wf = wave.open(self.file_name,'wb')
		wf.setnchannels(self.channels)
		wf.setsampwidth(self.port_audio.get_sample_size(self.format))
		wf.setframerate(self.rate)
		wf.writeframes(b''.join(self.sound_data))
		wf.close()
		
	def _sound_to_audio_data(self):	
		"""		
		Convert the recorded sound to an AudioData format
		This is needed for the recognition
		"""		
		audio_framerate = self.rate
		#Not sure about this one yet
		audio_smplwidth = 2		
		self.audio_data = sr.AudioData(b''.join(self.sound_data),sample_rate=audio_framerate,sample_width=audio_smplwidth)	
		
	def play_sound(self):
		winsound.PlaySound(self.file_name,winsound.SND_FILENAME)
		self.print_text('finished')
		
	def speech_to_text(self,language_type):				
		output_recognizer = self.recog.recognize_google(self.audio_data,language=language_type,show_all=True)
			
		conf_value = output_recognizer['alternative'][0]['confidence']
		text_value = output_recognizer['alternative'][0]['transcript']
		# Would love to be able to add more....
		self.display[0].text = text_value
		self.display[1].text = str(conf_value)
		
		text_file = open(re.sub('wav','txt',self.file_name), "w")
		text_file.write(text_value)
		text_file.close()
	
	def generate_news(self):
		if not self.build_init:
			self.build()

		self.random_article = np.random.randint(0,len(self.news_article))
		self.clean_text = CleanText([self.news_article[self.random_article]])
		self.print_main_text(self.news_article[self.random_article])

	def summarize(self,text,n=3):
		fs = FrequencySummarizer()
		sum_text = fs.summarize(text,n)
		self.print_main_text('\n '.join(sum_text))
	
	def clean_stopwords(self):
		self.print_main_text(self.clean_text.remove_stopwords_text().text[0])
		
	def clean_stemming(self):
		self.print_main_text(self.clean_text.stem_text().text[0])

	def clean_stopwords_and_stemming(self):
		self.print_main_text(self.clean_text.stem_text().remove_stopwords_text().text[0])		
			
	# subset_list(article)... could be done...
	
	# Use a variety of methods here..
	#article_clean = CleanText(article).remove_stopwords_article().remove_punc_article().stem_text().text

class SoundSummarizerApp(App):
	def build(self):
		test = SoundSummarizer()
		#text_thing = TextLayout()
		#test.add_widget(text_thing)
		return test
#test.add_widget(text_thing)



sumApp = SoundSummarizerApp()
sumApp.run()	