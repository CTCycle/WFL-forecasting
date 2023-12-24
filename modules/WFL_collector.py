import os
import sys
import time
from selenium import webdriver

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add modules path if this file is launched as __main__
#------------------------------------------------------------------------------
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import modules and components
#------------------------------------------------------------------------------
from modules.components.scraper_classes import WebDriverToolkit, XamigScraper
from modules.components.data_classes import PreProcessing
import modules.global_variables as GlobVar
import configurations as cnf

# [CONNECT TO XAMIG AND EXTRACT DATA]
#==============================================================================
# ...
#==============================================================================
print('''
-------------------------------------------------------------------------------
WIN FOR LIFE DATA COLLECTION    
-------------------------------------------------------------------------------
      
STEP 1 ---> Check chromedriver version
''')

# activate chromedriver and scraper
#------------------------------------------------------------------------------
modules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')
WD_toolkit = WebDriverToolkit(modules_path, GlobVar.data_path, headless=cnf.headless)
webdriver = WD_toolkit.initialize_webdriver()
webscraper = XamigScraper(webdriver, cnf.credentials, GlobVar.data_path)

# check if files downloaded in the past are still present, then remove them
#------------------------------------------------------------------------------
for filename in os.listdir(GlobVar.data_path):
    file_path = os.path.join(GlobVar.data_path, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# perform login to Xamig.com
#------------------------------------------------------------------------------
print('''STEP 2 ---> Extract data from Xamig.com archives   
''')

# perform login and download data
#------------------------------------------------------------------------------
webscraper.xamig_login(10)
years = list(range(2013, cnf.current_year + 1))
webscraper.WLF_data_extraction(5, years)
time.sleep(3)

# [ELABORATE DATA]
#==============================================================================
# ...
#==============================================================================
print('''STEP 3 ---> Raw data preparation   
''')

# generate list of names and paths
#------------------------------------------------------------------------------
download_files = [f for f in os.listdir(GlobVar.data_path) if os.path.isfile(os.path.join(GlobVar.data_path, f))]
file_paths = [os.path.join(GlobVar.data_path, x) for x in download_files if x.endswith('.txt')]

# extract text data
#------------------------------------------------------------------------------
PP = PreProcessing()
extractions = PP.dataset_composer(file_paths)

# add sum and mean of extractions
#------------------------------------------------------------------------------
number_cols = [f'N.{i+1}' for i in range(10)]
extractions[number_cols] = extractions[number_cols].astype('int32')
extractions['sum extractions'] = extractions[number_cols].sum(axis=1)
extractions['mean extractions'] = extractions[number_cols].mean(axis=1)

# save data as .csv
#------------------------------------------------------------------------------
file_loc = os.path.join(GlobVar.data_path, 'WFL_extractions.csv') 
extractions.to_csv(file_loc, index = False, sep = ';', encoding='utf-8')

print('''
-------------------------------------------------------------------------------
Data has been collected and saved as .csv in the designated folder  
-------------------------------------------------------------------------------
''')


