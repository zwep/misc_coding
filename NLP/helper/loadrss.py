# encoding: utf-8

"""
Here we show how to retrieve RSS feeds from some popular (Dutch) newswebsites

"""


class RssUrl:
    """
    RSS feeds for the news sites
    - NOS
    - NU
    - RTL

    Could add a lot more of course...
    """

    def __init__(self):
        """
        initializes the RssUrl.
        This allows you to easily create the urls to fetch data
        """
        self.news_source_list = ['nos', 'nu', 'rtl']
        self.nos_prefix = 'http://feeds.nos.nl/'
        self.nu_prefix = 'http://www.nu.nl/rss/'
        self.rtl_prefix = 'http://www.rtlnieuws.nl/service/rss/'
        self.rtl_suffix = '/index.xml'

        self.nos_label = ['nosnieuwsbinnenland', 'nossportalgemeen']
        self.nu_label = ['Algemeen', 'Sport']
        self.rtl_label = ['nederland', 'Sport']

    def add_label(self, news_source, new_label):
        """
        If you need to add a label later on... Not sure how usefull these are... since it is Python and not C or so

        :param news_source: 'nos' , 'nu', or 'rtl'
        :param new_label: New label that needs to be added...
        :return: None
        """
        if news_source == 'nos':
            self.nos_label.append(new_label)
        elif news_source == 'nu':
            self.nu_label.append(new_label)
        elif news_source == 'rtl':
            self.rtl_label.append(new_label)

    def get_label(self, news_source):
        """
        Returns the labels of the given newssource.. but again. Not too usefull when using Python.

        Should work on the similarity between some labels

        :param news_source: 'nos' , 'nu', or 'rtl'
        :return: the revelant labels for the news source
        """
        label = ""
        if news_source == 'nos':
            label = self.nos_label
        elif news_source == 'nu':
            label = self.nu_label
        elif news_source == 'rtl':
            label = self.rtl_label
        return label

    def get_url_feed(self, news_source):
        """
        Returns the full url-feed you need for the given type

        Hence, you will get all the labels..
        """
        url_feed = ""
        if news_source == 'nos':
            url_feed = [self.nos_prefix + x for x in self.nos_label]
        elif news_source == 'nu':
            url_feed = [self.nu_prefix + x for x in self.nu_label]
        elif news_source == 'rtl':
            url_feed = [self.rtl_prefix + x + self.rtl_suffix for x in self.rtl_label]
        else:
            print('Incorrect use of news_source, please use one of the following: ', ', '.join(self.news_source_list))

        return url_feed

    def get_url_content(self, news_source):
        """
        Returns the content of the called RSS Feed

        We might need this part in order to verify the validity of the request

        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent':user_agent}
        But I am not sure how to incorporate this into the feedparser.

        :param news_source: nos, rtl or nu
        :return: Beautiful soup object that contains all the text that is being returned..
        """
        import feedparser  # Usefull to parse RSS feeds
        from helper.miscfunction import dict_find
        from bs4 import BeautifulSoup as bS

        url_label = self.get_label(news_source)
        url_list = self.get_url_feed(news_source)
        rss_text = []
        for i_label, i_url in zip(url_label, url_list):
            url_content = feedparser.parse(i_url)
            rss_html_text = list(dict_find('value', url_content))
            for i_html in rss_html_text:
                process_html = bS(i_html, 'lxml').text.replace('\n', ' ')
                rss_text.append((news_source + '-' + i_label, process_html))
        return rss_text

    def get_all_content(self):
        """
        Returns all the content we can find at this moment

        :return: list of text and labels
        """
        full_text = []
        for i_news in self.news_source_list:
            full_text.extend(self.get_url_content(i_news))

        return full_text
