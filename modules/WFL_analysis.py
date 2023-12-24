import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from keras.utils.vis_utils import plot_model

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
from modules.components.data_classes import TimeSeriesAnalysis
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATASETS]
#==============================================================================
# Load patient dataset and dictionaries from .csv files in the dataset folder.
# Also, create a clean version of the exploded dataset to work on
#==============================================================================
filepath = os.path.join(GlobVar.data_path, 'WFL_extractions.csv')                
df_WFL = pd.read_csv(filepath, sep= ';', encoding='utf-8')


print(f'''
-------------------------------------------------------------------------------
WFL Analysis
-------------------------------------------------------------------------------
Leverage large volume of Win For Life extractions (from the Xamig.com database)
and predict future extractions based on the observed timeseries 
''')

# [COLOR MAPPING AND ENCODING]
#==============================================================================
# ...
#==============================================================================
print(f'''STEP 1 -----> Analysize timeseries
''')

analyzer = TimeSeriesAnalysis()
number_cols = [f'N.{i+1}' for i in range(10)]
stats_cols = ['sum extractions', 'mean extractions']
info_cols = ['Concorso', 'Data', 'Ora']

# histograms
#------------------------------------------------------------------------------
analyzer.timeseries_histogram(df_WFL[number_cols], 11, GlobVar.als_path)