# encoding: utf-8

"""

derp

"""


from moviepy.editor import *

fullvideo = VideoFileClip("record_test.avi")
vid1 = fullvideo.subclip(19, 32)
vid2 = fullvideo.subclip(35, 50)
vid3 = fullvideo.subclip(62, 90)
vidclips = [vid1, vid2, vid3]
# Make the text. Many more options are available.
# txt_clip = ( TextClip("My Holidays 2013",fontsize=70,color='white')
#              .set_position('center')
#              .set_duration(10) )


final_clip = concatenate_videoclips(vidclips)

# You can write any format, in any quality.
final_clip.write_videofile("final.mp4", bitrate="5000k")

# result = CompositeVideoClip([video, txt_clip]) # Overlay text on video
# result.write_videofile("myHolidays_edited.webm",fps=25) # Many options....
#
#
# concatenate_videoclips