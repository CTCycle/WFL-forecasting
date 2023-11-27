import os
import time
from tqdm import tqdm
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class WebDriverToolkit:    
    
    def __init__(self, driver_path, download_path, headless=True):
        self.driver_path = os.path.join(driver_path, 'chromedriver.exe') 
        self.download_path = download_path      
        self.option = webdriver.ChromeOptions()
        if headless==True:
            self.option.add_argument('--headless')
        self.service = Service(executable_path=self.driver_path)
        self.chrome_prefs = {'download.default_directory' : download_path}
        self.option.experimental_options['prefs'] = self.chrome_prefs
        self.chrome_prefs['profile.default_content_settings'] = {'images': 2}
        self.chrome_prefs['profile.managed_default_content_settings'] = {'images': 2}

    def initialize_webdriver(self):       
        self.path = ChromeDriverManager().install()
        self.service = Service(executable_path=self.path)
        driver = webdriver.Chrome(service=self.service, options=self.option)
        print('The latest ChromeDriver version is now installed in the .wdm folder in user/home')                  
        
        return driver 


    
# define the class for inspection of the input folder and generation of files list
#==============================================================================
#==============================================================================
#==============================================================================
class XamigScraper: 

    def __init__(self, driver, credentials, download_path):         
        self.driver = driver
        self.dw_path = download_path
        self.username = credentials['username']
        self.password = credentials['password']
        self.login_URL = 'https://www.xamig.com/login.php'       
        
               
    #==========================================================================
    def xamig_login(self, wait_time):  
        self.driver.get(self.login_URL)
        wait = WebDriverWait(self.driver, wait_time)        
        username_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="user"]')))
        username_input.send_keys(self.username)
        password_input = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="pass"]')))
        password_input.send_keys(self.password)   
        button = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="loginform"]/button')))
        button.click() 
        accept = wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[2]/div[2]/div[1]/div[2]/div[2]/button[1]')))
        accept.click() 
    
    
    #==========================================================================
    def WLF_data_extraction(self, wait_time, years):

        wait = WebDriverWait(self.driver, wait_time)             
        for year in tqdm(years):
            per_year_URL = f'https://www.xamig.com/win-for-life/esporta-archivio-pdf-txt-xls.php?anno={year}'
            self.driver.get(per_year_URL)                
            item = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="format2"]')))
            item.click()
            item = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="estr"]/div[2]/div[2]/input')))
            item.click() 
            while any([filename.endswith('.crdownload') for filename in os.listdir(self.dw_path)]):
                time.sleep(2) 
         
       
        
        
 
    

 