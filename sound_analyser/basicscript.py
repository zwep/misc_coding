import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "apt.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
			channels=CHANNELS,
			rate=RATE,
			input=True,
			frames_per_buffer=CHUNK)

print("* Recording audio...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	data = stream.read(CHUNK)
	frames.append(data)


print("* done\n" )

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


#-------------

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from pydub import AudioSegment


file_name_mp3 = '/home/charmmaria/Music/popping.mp3'
file_name = '/home/charmmaria/Music/poppibg.m4a'
# spf = wave.open('apt.wav','r')
spf = wave.open(file_name_mp3, 'r')
test = AudioSegment.from_file(file_name)
test = AudioSegment.from_mp3(file_name)
dir(test)
len(test)
test.duration_seconds
len(test.get_array_of_samples())
len(test.raw_data)
z = np.array(test.get_array_of_samples())
plt.plot(z)
#spf wave.open('20170804_sound_record.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')


#If Stereo
if spf.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('Signal Wave...')
plt.plot(signal)
plt.show()