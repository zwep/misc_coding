#------ getting all the leaderboards

import re
from selenium import webdriver
# Used for drag and drop
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome()
query_url = "https://social.connect.abnamro.com/search/web/search?scope=profiles"
driver.get(query_url)
driver.execute_script("searchObject.setPageSize(100)")

# Add some more code such that no people are excluded...

max_page_elem = driver.find_element_by_id("contentContainer_results_View_CenterPaging")
max_page = int(re.sub(".*\\.([0-9]+)Next","\\1",max_page_elem.text))

full_pictures = []
full_names = []
for i_index in range(1,max_page+1):
	print("PAGE: ",i_index)
	driver.execute_script("searchObject.performPagination(" + str(i_index) + ")")
	time.sleep(2)
	url_elem = driver.find_element_by_class_name('lconnProfilesPhotoContainer')
	link_to_urls = url_elem.find_elements_by_xpath("//img")
	picture_urls = [x.get_attribute("src") for x in link_to_urls]
	# Filter on only the useful ones
	picture_urls = [x for x in picture_urls if bool(re.search("guid",x))]
	name_elements = driver.find_elements_by_class_name('hasHover')
	people_name = [x.text for x in name_elements]
	print(len(people_name),len(picture_urls))

# Problem that keeps occuring...
# - element staleness.. has to do with reloading of the page -> This seems to be fixed now
# - !!some pages containt data quality issue (mismatch between images and names)

# Todo...
# - images are not extracted yet
# - location to store the images needs to be created
# - initial start up (not excluding any people) is still missing  

# --- Code to make a screenshot and save as a file
website_connections_foto = "https://social.connect.abnamro.com/profiles/photo.do?guid=225739"
page.get(website_connections_foto)
page.get_screenshot_as_file('C:/Users/C35612.LAUNCHER/derp.png')

