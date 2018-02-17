
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




