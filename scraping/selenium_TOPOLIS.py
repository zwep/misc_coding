
http://selenium-python.readthedocs.io/getting-started.html#simple-usage


from selenium import webdriver
# Used for drag and drop
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome()
driver.get("http://www.python.org")

website_topolis = "https://www.topoliscity.com/"

quest_5 = "https://www.topoliscity.com/timeline/quest/160"
question_51 = "/intervention/8411/question/" 
question_51_list = [quest_5 + question_51 + str(i) for i in range(2,6)]
  
temp = question_5_list[0]
driver.get(temp)

content = driver.find_element_by_class_name('is-correct-answer is-selected')
content.click()
# Search for correct answer 


question_52 = "/intervention/8411/question/" 
question_52_list = [quest_5 + question_52 + str(i) for i in [3,5,7,9]]
temp = question_52_list[0]

driver.get(temp)

content = driver.find_element_by_class_name('btn btn-appleblue active')

# Or check data-answer-index = 1
# How to move objects with selenium?
source_element = driver.find_element_by_name('your element to drag')
dest_element = driver.find_element_by_name('element to drag to')
ActionChains(driver).drag_and_drop(source_element, dest_element).perform()

quest_6 = "https://www.topoliscity.com/timeline/quest/162"


quest_7 = "https://www.topoliscity.com/timeline/quest/689"


quest_8 = "https://www.topoliscity.com/timeline/quest/720"


quest_9 = "https://www.topoliscity.com/timeline/quest/742"


quest_10 = "https://www.topoliscity.com/timeline/quest/762"




#------ getting all the leaderboards

from selenium import webdriver
# Used for drag and drop
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Chrome()


website_topolis = "https://www.topoliscity.com/leaderboard"
driver.get(website_topolis)

content = driver.find_element_by_class_name('grid-canvas')



#------ getting all the leaderboards
# IE explorer


from selenium import webdriver
# Used for drag and drop
from selenium.webdriver.common.action_chains import ActionChains

driver = webdriver.Ie()
page = webdriver.Ie(capabilities={'ignoreZoomSetting':True})

website_topolis = "https://www.topoliscity.com/leaderboard"
page.get(website_topolis)

content = driver.find_element_by_class_name('grid-canvas')


# For getting connections
website_connections_A = "https://social.connect.abnamro.com/search/web/search?scope=profiles&query=A#%3Fscope%3Dprofiles%26query%3DA*%26page%3D1%26pageSize%3D10%26personalization%3D%7B%22type%22%3A%22personalContentBoost%22%2C%22value%22%3A%22on%22%7D"
page.get(website_connections_A)
https://social.connect.abnamro.com/search/web/search?scope=profiles&query=A#%3Fscope%3Dprofiles%26query%3DA*%26page%3D1%26pageSize%3D10%26personalization%3D%7B%22type%22%3A%22personalContentBoost%22%2C%22value%22%3A%22on%22%7D

website_connections_foto = "https://social.connect.abnamro.com/profiles/photo.do?guid=225739"
page.get(website_connections_foto)
page.get_screenshot_as_file('C:/Users/C35612.LAUNCHER/derp.png')

#urllib.request.urlretrieve(website_connections_foto, "captcha.png")