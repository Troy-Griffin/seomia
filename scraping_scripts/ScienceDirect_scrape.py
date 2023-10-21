import scrapy

from selenium import webdriver
#from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
import pandas as pd
import time

import scrapy
from scrapy.crawler import CrawlerProcess
import json

if __name__ == "__main__":
    url = 'https://innopac.wits.ac.za/wamvalidate?url=https%3A%2F%2F0-www-sciencedirect-com.innopac.wits.ac.za%3A443%2Fsearch%2Fadvanced#'
    #service = Service("geckodriver.exe")
    #driver = webdriver.Firefox(service=service)
    driver = webdriver.Chrome()
    driver.maximize_window()
    handle = driver.current_window_handle
    driver.get(url)
    action = webdriver.ActionChains(driver)
    
    driver.implicitly_wait(5)
    wait = WebDriverWait(driver, 60)

    name = 'Griffin'
    number = 2131783

    wait.until(EC.visibility_of_element_located((By.XPATH, "//input[@name='name']"))).send_keys(name)
    wait.until(EC.visibility_of_element_located((By.XPATH, "//body[@class='bodybg']/div[@class='minHeight']/div[@class='fullPage']/div[@class='pageContent']/div[@class='pageContentInner']/div[@id='accessibleForm']/form[@method='post']/fieldset/div[2]/input[1]"))).send_keys(number)
    
    wait.until(EC.visibility_of_element_located((By.XPATH, "//body[@class='bodybg']/div[@class='minHeight']/div[@class='fullPage']/div[@class='pageContent']/div[@class='pageContentInner']/div[@id='accessibleForm']/form[@method='post']/fieldset/div[3]/input[1]"))).send_keys(number)
    
    wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@class='formButtonArea']//div[@class='formButtonArea']//a[@href='#']"))).click()
    
    search = 'mechanical aeronautical industrial engineering'
    wait.until(EC.visibility_of_element_located((By.XPATH, "//textarea[@id='qs']"))).send_keys(search)
    
    #action.send_keys(Keys.ENTER)
    wait.until(EC.visibility_of_element_located((By.XPATH, "//div[@class='BottomLinksGroup']//div[@class='row search-row']//div[@class='col-xs-24 col-sm-20 col-lg-12']//button[@type='submit']"))).click()
    
    wait.until(EC.visibility_of_element_located((By.XPATH, "//body[@class='js']/div[@id='react-root']/div[@class='sd-flex-container']/div[@class='sd-flex-content']/div[@class='SearchPage']/div[@class='grid']/div[@class='row']/div[@class='col-xs-24']/section/div[@class='Search']/div[@id='main_content']/div[@id='srp-facets']/div[@class='facet-container']/form[@name='filters']/div/div[2]/fieldset[1]/ol[1]/li[2]/div[1]/label[1]/span[1]"))).click()
    wait.until(EC.visibility_of_element_located((By.XPATH, "//body[@class='js']/div[@id='react-root']/div[@class='sd-flex-container']/div[@class='sd-flex-content']/div[@class='SearchPage']/div[@class='grid']/div[@class='row']/div[@class='col-xs-24']/section/div[@class='Search']/div[@id='main_content']/div[@id='srp-facets']/div[@class='facet-container']/form[@name='filters']/div/div[5]/fieldset[1]/ol[1]/li[1]/div[1]/label[1]/span[1]")))
    checkbox1 = driver.find_element(By.XPATH, "//body[@class='js']/div[@id='react-root']/div[@class='sd-flex-container']/div[@class='sd-flex-content']/div[@class='SearchPage']/div[@class='grid']/div[@class='row']/div[@class='col-xs-24']/section/div[@class='Search']/div[@id='main_content']/div[@id='srp-facets']/div[@class='facet-container']/form[@name='filters']/div/div[5]/fieldset[1]/ol[1]/li[1]/div[1]/label[1]/span[1]")
    driver.execute_script("arguments[0].click();", checkbox1)
    
    wait.until(EC.visibility_of_element_located((By.XPATH, "//body[@class='js']/div[@id='react-root']/div[@class='sd-flex-container']/div[@class='sd-flex-content']/div[@class='SearchPage']/div[@class='grid']/div[@class='row']/div[@class='col-xs-24']/section/div[@class='Search']/div[@id='main_content']/div[@class='SearchBody row transparent']/div[@class='transparent results-container col-xs-24 col-sm-16 col-lg-18 visible']/div/div[@id='srp-pagination-options']/div[@class='move-left']/ol[@class='ResultsPerPage hor-separated-list prefix suffix']/li[3]/a[1]")))
    checkbox2 = driver.find_element(By.XPATH, "//body[@class='js']/div[@id='react-root']/div[@class='sd-flex-container']/div[@class='sd-flex-content']/div[@class='SearchPage']/div[@class='grid']/div[@class='row']/div[@class='col-xs-24']/section/div[@class='Search']/div[@id='main_content']/div[@class='SearchBody row transparent']/div[@class='transparent results-container col-xs-24 col-sm-16 col-lg-18 visible']/div/div[@id='srp-pagination-options']/div[@class='move-left']/ol[@class='ResultsPerPage hor-separated-list prefix suffix']/li[3]/a[1]")
    driver.execute_script("arguments[0].click();", checkbox2)
    time.sleep(5)
    
    wait.until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='result-item-content']")))
    articles = driver.find_elements(By.CLASS_NAME, "result-item-content")
    
    df = pd.DataFrame({'Title': [], 'Abstract': [], 'Subject': [], 'Journal': []})
    try:
        k=0
        while k==0:
            wait.until(EC.visibility_of_element_located((By.XPATH, "//li[@class='pagination-link next-link']//a[@class='anchor']")))
            next = driver.find_element(By.XPATH, "//li[@class='pagination-link next-link']//a[@class='anchor']")
            for article in articles:
                subject =[]
                
                title = article.find_element(By.XPATH, ".//span[@class='anchor-text']/span").text
                
                journal = article.find_element(By.XPATH, ".//a[@class='anchor subtype-srctitle-link anchor-default anchor-has-inherit-color']/span/span").text

                btn = article.find_element(By.XPATH, ".//span[@class='PreviewButton']/button")
                driver.execute_script("arguments[0].click();", btn)
                action.moveToElement(btn).perform()
                try:
                    wait.until(EC.visibility_of_element_located((By.XPATH, ".//div[@class='abstract-section u-font-serif']/p")))
                    abstract = article.find_element(By.XPATH, ".//div[@class='abstract-section u-font-serif']/p").text
                except:
                    abstract = "NA"
                
                if 'mech' in title:
                    subject.insert(0, 'mechanical')
                if 'aero' in title:
                    subject.insert(0, 'aeronautical')
                if 'indus' in title:
                    subject.insert(0, 'industrial')
                if subject == []:
                    subject.insert(0, 'misc')
        
                row = {'Title': title, 'Abstract': abstract, 'Subject': subject, 'Journal': journal}
                df = df._append(row, ignore_index=True)
            
            if next is not None:
                driver.execute_script("arguments[0].click();", next)
            else:
                k=1
            wait.until(EC.visibility_of_all_elements_located((By.XPATH, "//div[@class='result-item-content']")))
            articles = driver.find_elements(By.XPATH, "//li[@class='ResultItem col-xs-24 push-m']")  

        path = "Data.xlsx"
        df.to_excel(path, sheet_name="Data", index=False)
    except:
        path = "Data.xlsx"
        df.to_excel(path, sheet_name="Data", index=False)