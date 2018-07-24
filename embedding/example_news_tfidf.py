# encoding: utf-8

"""
Here we show

"""

from helper.loadrss import RssUrl
from helper.scorefunction import score_logistic_regression
from helper.miscfunction import transform_to_color

from textprocessing.processtext import CleanText

from embedding.docembedding import TFIDF


"""
Define functions
"""

if __name__ == '__main__':
    A = RssUrl()
    article, article_label = A.get_all_content()
    article_label_color = transform_to_color(article_label)

    # Extra articles that can be checked
    nos_nieuws = ['binnenland', 'algemeen', 'buitenland', 'politiek', 'economie', 'opmerkelijk', 'koningshuis',
                  'cultuurenmedia', 'tech']
    nos_nieuws = ['nosnieuws' + x for x in nos_nieuws]
    nos_sport = ['algemeen', 'voetbal', 'wielrennen', 'schaatsen', 'tennis']
    nos_sport = ['nossport' + x for x in nos_sport]
    nos_overig = ['nieuwsuuralgemeen', 'jeugdjeugdjournaal']
    nu_nieuws = ['Algemeen', 'Economie', 'Internet', 'Sport', 'Achterklap', 'Opmerkelijk', 'Muziek', 'Film', 'Boek',
                 'Games', 'Wetenschap', 'Gezondheid']
    rtl_nieuws = ['nederland', 'nederland/politiek', 'buitenland', 'opmerkelijk', 'gezin', 'gezondheid', 'technieuws']
    rtl_sport = ['algemeen', 'voetbal']
    rtl_boulevard = ['entertainment', 'lifestyle', 'crime', 'royalty']

    # Use a variety of methods here..
    article_clean = CleanText(article).remove_stopwords_text().remove_punc_text().stem_text().text

    tf_idf = TFIDF(article_clean)
    # USe a variety of methods here
    tf_idf_features = tf_idf .TF_count
    tf_idf_label = list(tf_idf .vocab_dict.keys())
    score_logistic_regression(tf_idf_features, article_label, tf_idf_label, article_label_color)

# my_url = 'http://www.rtlnieuws.nl/nederland/doe-het-zelf-grafkist-zet-je-in-20-minuutjes-in-elkaar'

# The assembled request
# request = urllib.request.Request(my_url,None,headers) 
# response = urllib.request.urlopen(request)
# data = response.read() # The data u need
# z = BeautifulSoup(data,'html.parser')
# derp = z.find_all('div',{'class':'paragraph'})
# lala = derp.findChildren()
# Now we have all the children.. which is a list..
# This list can then be used to really get the right text.
