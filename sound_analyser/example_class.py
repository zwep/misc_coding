"""
Defining function
"""

a = wav_to_audio_data('2017-10-02 112625925745.wav')

recog = sr.Recognizer()

output_recognizer = recog.recognize_google(a, language='nl-NL', show_all=True)
output_recognizer = recog.recognize_google(a, language='nl-US', show_all=True)

loc_sound = 'D:\data\Production_data\Sound_record'


# =============
#  Examples
# ============

# current_date = pd.datetime.today()
# current_date_str = str(current_date)[0:10]

# derp = SoundFile(current_date_str)

# os.chdir(loc_sound)
# derp.record_sound(5)
# derp.plot_sound()
# derp.write_sound()


