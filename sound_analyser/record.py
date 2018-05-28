# encoding: utf-8

"""

This file is used to record audio by simply executing this script and then saving the recording with a unique name.
- Record Audio
- Save under unique name
- Under any circumstance record the audio
- Record for a (now) fixed amount of seconds

Future:
- Can be made into a function

I think this has some nice overview of the API
http://portaudio.com/docs/v19-doxydocs/api_overview.html

"""

import pyaudio
import wave

import numpy as np
import matplotlib.pyplot as plt

import winsound
import speech_recognition as sr


"""
Defining a recdoring class
"""


class SoundFile:
    """
    Here we define a class to record, pay, read and write data
    """
    def __init__(self, file_name, chunk_size=2014, format=pyaudio.paInt16, channels=1, rate=44100):
        # initialize variables
        self.chunk = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.file_name = file_name + ".wav"

        # initialize classes
        self.port_audio = pyaudio.PyAudio()
        self.record_seconds = int()
        self.sound_data = []
        self.audio_data = sr.AudioData(b'', 1, 1)

    @staticmethod
    def wav_to_audio_data(fl_name):
        """
        Convert the recorded sound to an AudioData format
        This is needed for the recognition

        :param fl_name: file name that needs to be converted
        """
        spf = wave.open(fl_name, 'r')

        audio_framerate = spf.getframerate()
        audio_smplwidth = spf.getsampwidth()

        # Extract Raw Audio from Wav File
        signal_bit = spf.readframes(-1)
        # Not returning singal_int for now.,,.
        spf_audio = sr.AudioData(signal_bit, sample_rate=audio_framerate, sample_width=audio_smplwidth)

        return spf_audio

    def print_device(self):
        """
        Here we show the devices that are current plugged in
        Set the channels variable in order to records from the right output
        """

        self.port_audio.get_host_api_count()  # Get the amount of host_apis
        info = self.port_audio.get_host_api_info_by_index(0)  # What about the other indeces?
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            something_info = self.port_audio.get_device_info_by_host_api_device_index(0, i)
            if (something_info.get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", self.port_audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    def record_sound(self, record_seconds):
        """
        With the inialized variables... records a sound
        """
        stream = self.port_audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        # WHy do we capture d = [] here while write data...
        # And what is the differennce between s.write() and wf.write()?

        print("* recording")
        for i in range(0, (self.rate // self.chunk * record_seconds)):
            data = stream.read(self.chunk)
            self.sound_data.append(data)
        print("* done")
        stream.stop_stream()
        stream.close()

        # convert the just recorded sound to AudioData
        self._sound_to_audio_data()

    def plot_sound(self):
        """
        Plotting the sound data that has just been generated
        """
        # Could add some option that first checks the length of sound_data..
        fig = plt.figure()
        s = fig.add_subplot(111)
        amplitude = np.fromstring(b''.join(self.sound_data), np.int16)
        s.plot(amplitude)
        plt.show()

    def speech_to_text(self):
        output_recognizer = r.recognize_google(self.audio_data,language='nl-NL', show_all=True)
        conf_value = output_recognizer['alternative'][0]['confidence']
        text_value = output_recognizer['alternative'][0]['transcript']
        # Would love to be able to add more....
        return text_value, conf_value

    def write_sound(self, file_name=''):
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

    def play_sound(self):
        winsound.PlaySound(self.file_name,winsound.SND_FILENAME)

    def _sound_to_audio_data(self):
        """
        Convert the recorded sound to an AudioData format
        This is needed for the recognition
        """
        audio_framerate = self.rate
        # Not sure about this one yet
        audio_smplwidth = 2
        self.audio_data = sr.AudioData(self.sound_data,sample_rate=audio_framerate,sample_width=audio_smplwidth)


