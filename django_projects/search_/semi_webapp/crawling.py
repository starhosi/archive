from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd
import numpy as np
import time
class s_info:
    def __init__(self,store,score,review,lin):
        self.store = store
        self.score = score
        self.review = review
        self.lin = lin

def crawling_result(keyword):
    driver = webdriver.Chrome('/usr/local/bin/chromedriver')
    url= 'https://map.kakao.com/'
    driver.get(url)

    loc_name = keyword[0]
    food_name = keyword[1]

    driver.find_element_by_name('q').send_keys(loc_name+food_name)
    driver.find_element_by_xpath("//button[@id='search.keyword.submit']").send_keys(Keys.ENTER)
    time.sleep(1)
    time.sleep(1)
    qwer=driver.find_elements_by_class_name('label')
    qwer[3].send_keys(Keys.ENTER)
    time.sleep(1)
    
    # 1) 가게 이름
    store_name = driver.find_elements_by_class_name('link_name')
    store_list = []
    for i in store_name:
        store_list.append(i.text)
    # 2) 평점
    score = driver.find_elements_by_class_name('num')
    score_list = []
    for i in score:
        score_list.append(i.text)
    # 3) 리뷰
    review = driver.find_elements_by_class_name('review')
    review_list=[]
    for i in review:
        s = i.text
        review_list.append(int(s[3:]))

    # 4) 링크
    store_link = 'https://www.google.com/search?q='
    slink_lst = []
    for i in store_list:
        store_link = 'https://www.google.com/search?q='
        store_link = store_link + i
        store_link = store_link.replace(' ','+')
        slink_lst.append(store_link)

    driver.close()
    driver.quit()

    score_list=[e for e in score_list if e !='' ]
    df_accuracy = pd.DataFrame({'Store':store_list, 'Score':score_list, 'Review':review_list})

    result_list=[]
    for i in range(15):
        result_list.append([store_list[i], score_list[i], review_list[i],slink_lst[i]])
    result_list = sorted(result_list,key=lambda x:x[2],reverse=True)

    r_lst = []
    for a in result_list:
        r_lst.append(s_info(a[0],a[1],a[2],a[3]))

    # print(result_list[0].store)
    return r_lst
