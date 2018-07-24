# encoding: utf-8

"""
Here we can showcase the frequencysummarizer.

"""
from summarization.frequencysummarizer import FrequencySummarizer

news_to_summarize = 'NEW DELHI – The Dalai Lama has spoken out for the first time about the Rohingya refugee crisis, ' \
                    'saying Buddha would have helped Muslims fleeing violence in Buddhist-majority Myanmar. Hundreds ' \
                    'of thousands of Rohingya have arrived in Bangladesh in recent weeks after violence flared in ' \
                    'neighboring Myanmar, where the stateless Muslim minority has endured decades of persecution. ' \
                    'The top Buddhist leader is the latest Nobel Peace Prize laureate to speak out against the ' \
                    'violence, which the U.N. special rapporteur on human rights in Myanmar says may have killed ' \
                    'more than 1,000 people, most of them Rohingya. “Those people who are sort of harassing some ' \
                    'Muslims, they should remember Buddha,” the Dalai Lama told journalists who asked him about the ' \
                    'crisis on Friday evening. “He would definitely give help to those poor Muslims. So still I feel ' \
                    'that. So very sad.” Myanmar’s population is overwhelmingly Buddhist and there is widespread ' \
                    'hatred for the Rohingya, who are denied citizenship and labeled illegal “Bengali” immigrants. ' \
                    'Buddhist nationalists, led by firebrand monks, have operated a long Islamophobic campaign ' \
                    'calling for them to be pushed out of the country. Myanmar’s de facto civilian leader, ' \
                    'Aung San Suu Kyi, has been condemned for her refusal to intervene in support of the Rohingya, ' \
                    'including by fellow Nobel laureates Malala Yousafzai and Desmond Tutu. Archbishop Tutu, ' \
                    'who became the moral voice of South Africa after helping dismantle apartheid there, ' \
                    'last week urged her to speak out. “If the political price of your ascension to the highest ' \
                    'office in Myanmar is your silence, the price is surely too steep,” Tutu said in a statement.'

review_to_summarize = "As usual, the critics fail to grasp the obvious. Does the movie entertain? Yes, " \
                 "it does. Absolutely. Sure, it may not match the real story, and many things are not based in " \
                      "reality, but that's the point! It's a movie! It is a fun musical, very well done and " \
                      "enjoyable. If it was a made up story about John G. Pigglestack, then the critics would have " \
                      "nothing to complain about. I actually likes this much better than La La Land. Guess it's more " \
                      "upbeat. Musicals are a rare breed. Enjoy them while you can."

tweet_to_summarize = "Bad ratings \@CNN & \@MSNBC got scammed when they covered the anti-Trump Russia rally " \
                     "wall-to-wall. They probably knew it was Fake News but, because it was a rally against me, " \
                     "they pushed it hard anyway. Two really dishonest newscasters, but the public is wise!"


fs = FrequencySummarizer()
print(fs.summarize(news_to_summarize, 2))
print(fs.summarize(review_to_summarize, 2))
print(fs.summarize(tweet_to_summarize, 2))
print(tweet_to_summarize)

