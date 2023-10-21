from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
import pandas as pd

if __name__ == "__main__":
    url = 'http://www.libraryresearch.com/'
    #service = Service("geckodriver.exe")
    #driver = webdriver.Firefox(service=service)
    driver = webdriver.Chrome()
    driver.maximize_window()
    handle = driver.current_window_handle
    driver.get(url)
    
    
    driver.implicitly_wait(5)
    wait = WebDriverWait(driver, 60)

    wait.until(EC.visibility_of_element_located((By.XPATH, "//button[contains(@class,'osano-cm-dialog__close osano-cm-close')]")))
    database_button = driver.find_element(By.XPATH, "//button[contains(@class,'osano-cm-dialog__close osano-cm-close')]")
    if database_button is not None:
        database_button.click()

    search = '(mechanical engineering) OR (aeronautical engineering) OR (industrial engineering) OR (materials science)'

    wait.until(EC.visibility_of_element_located((By.XPATH, "//button[contains(text(),'Advanced search')]")))
    search_button = driver.find_element(By.XPATH, "//button[contains(text(),'Advanced search')]")
    if search_button is not None:
       search_button.click()
    
    wait.until(EC.visibility_of_element_located((By.XPATH, "/html[1]/body[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/div[2]/main[1]/div[1]/div[1]/div[1]/div[1]/form[1]/div[1]/div[1]/div[1]/fieldset[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]")))
    search_entry = driver.find_element(By.XPATH, "/html[1]/body[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[2]/div[2]/main[1]/div[1]/div[1]/div[1]/div[1]/form[1]/div[1]/div[1]/div[1]/fieldset[1]/div[1]/div[1]/div[1]/div[1]/div[1]/input[1]")
    if search_entry is not None:
        search_entry.send_keys(search)
        

    initiate_search = driver.find_element(By.XPATH, "//button[@type='submit']")
    if initiate_search is not None:
        initiate_search.click()

    k=0
    while k == 0:
        wait.until(EC.visibility_of_element_located((By.XPATH, "//button[@data-auto = 'show-more-button']")))
        
        button = driver.find_element(By.XPATH, "//button[@data-auto = 'show-more-button']")
        if button is not None:
            button.location_once_scrolled_into_view
            driver.execute_script("arguments[0].click();", button)
        else:
            k=1
    
    links =[]
    articles = driver.get_elements(By.XPATH, "//a[@class='result-item-title__link']")
    for entry in articles:
        href = entry.get_attribute("href")
        links.insert(-1, str(url + href))
    
    data_final = {}
    for link in links:
        driver.get(link)
        headings = driver.get_elements(By.XPATH, "//article[@lang='en']//h3")
        data = driver.get_elements(By.XPATH, "//article[@lang='en']//ul")
        authors = []
        subjects = []
        author_affiliations = []
        legend = {}
        for i, entry in enumerate(data):
            data_points = entry.get_elements(By.XPATH, "//li")
            variable = headings[i].getText()
            values = []
            for point in data_points:
                values.insert(-1, point.getText())
            legend[variable] = values
        data_final[link] = legend


