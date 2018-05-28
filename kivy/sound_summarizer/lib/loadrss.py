from bs4 import BeautifulSoup
import feedparser

class RssUrl:
	def __init__(self):
		"""
		initializes the RssUrl.
		This allows you to easily create the urls to fetch data
		"""
		self.nos_prefix = 'http://feeds.nos.nl/'
		self.nu_prefix = 'http://www.nu.nl/rss/'
		self.rtl_prefix = 'http://www.rtlnieuws.nl/service/rss/'
		self.rtl_suffix = '/index.xml'
		
		self.nos_label = ['nosnieuwsbinnenland','nossportalgemeen']
		self.nu_label = ['Algemeen','Sport']
		self.rtl_label = ['nederland','Sport']
	
	def add_label(self,type,new_label):
		"""
		If you need to add a label later on...
		"""
		if type == 'nos':
			self.nos_label.append(new_label)
		elif type == 'nu':
			self.nu_label.append(new_label)
		elif type == 'rtl':
			self.rtl_label.append(new_label)
			
	def get_label(self,type):
		"""
		Returns the labels, useful for classification
		Should work on the similarity between some labels
		"""
		label = ""
		if type == 'nos':
			label = self.nos_label
		elif type == 'nu':
			label = self.nu_label
		elif type == 'rtl':
			label = self.rtl_label
		return label
			
	def get_url_feed(self,type):
		"""
		Returns the full url-feed you need for the given type
		Hence, you will get all the labels..
		"""
		url_feed = ""
		if type == 'nos':
			url_feed = [self.nos_prefix + x for x in self.nos_label]
		elif type == 'nu':
			url_feed = [self.nu_prefix + x for x in self.nu_label]
		elif type == 'rtl':
			url_feed = [self.rtl_prefix + x + self.rtl_suffix for x in self.rtl_label]
		return url_feed

		
# Used to find the values of a certain key in a nested dict
def dict_find(key, dictionary):
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in dict_find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in dict_find(key, d):
                    yield result	
					
# Comment about the feeds:
# They do not give the full set of text... there is an option to get 
# the url and fetch the full html text. But this is extra work.
# Could be done later.
def get_url_content():
	"""
	For some fixed labels per news-site, return the content.
	Might not be as dynamical as initially thought. But it serves a purpose:
	Easy access to news data. Of course, could add more variables.
	"""
	url_list = RssUrl()
	news_list = ['nos','nu','rtl']
	news_list = ['nos']
	print("loading data for: "+', '.join(news_list))
	url_content = []
	url_content_category = []
	for i_news in news_list:
		i_news_url = url_list.get_url_feed(i_news)	
		for i_index,i_url in enumerate(i_news_url):
			i_label = url_list.get_label(i_news)[i_index]
			print(i_url,i_label)
			d = feedparser.parse(i_url)
			A = list(dict_find('value',d))
			for A_1 in A:
				process_html = BeautifulSoup(A_1,'lxml').text.replace('\n',' ')
				url_content.append(process_html)
				url_content_category.append(i_label)
	return url_content, url_content_category
	