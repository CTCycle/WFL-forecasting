import os
import sys
import numpy as np
import pandas as pd
import pickle

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
from modules.components.data_classes import PreProcessing
from modules.components.training_classes import ModelTraining
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD MODEL AND DATA]
#==============================================================================
# ....
#==============================================================================        
print(f'''
-------------------------------------------------------------------------------
FAIRS predictions
-------------------------------------------------------------------------------
...
''')

# Load dataset of prediction inputs (if the file is present in the target folder)
# else creates a new csv file named predictions_inputs.csv
#------------------------------------------------------------------------------
if 'predictions_input.csv' not in os.listdir(GlobVar.fc_path):
    filepath = os.path.join(GlobVar.data_path, 'WFL_extractions.csv')                
    df_predictions = pd.read_csv(filepath, sep= ';', encoding='utf-8') 
    df_predictions = df_predictions[-cnf.predictions_size:]
else:
    filepath = os.path.join(GlobVar.pp_path, 'predictions_inputs.csv')                
    df_predictions = pd.read_csv(filepath, sep= ';', encoding='utf-8')

# drop unused columns and reset index
#------------------------------------------------------------------------------
df_predictions.drop(columns=['sum extractions', 'mean extractions'], inplace=True)
df_predictions.reset_index(inplace=True)

# Load model
#------------------------------------------------------------------------------
trainworker = ModelTraining(device = cnf.training_device) 
model = trainworker.load_pretrained_model(GlobVar.model_path)
load_path = trainworker.model_path
parameters = trainworker.model_configuration
model.summary(expand_nested=True)

# Load scikit-learn normalizer and one-hot encoders
#------------------------------------------------------------------------------
encoder_path = os.path.join(load_path, 'preprocessed data', 'OHE_encoder_extractions.pkl')
with open(encoder_path, 'rb') as file:
    encoder_ext = pickle.load(file)
encoder_path = os.path.join(load_path, 'preprocessed data',  'OHE_encoder_special.pkl')
with open(encoder_path, 'rb') as file:
    encoder_sp = pickle.load(file)
encoder_path = os.path.join(load_path, 'preprocessed data',  'normalizer.pkl')
with open(encoder_path, 'rb') as file:
    normalizer = pickle.load(file)

# [MAPPING AND ENCODING]
#==============================================================================
# ...
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Data preprocessing
-------------------------------------------------------------------------------

''')

# define subsets of columns to split datasets
#------------------------------------------------------------------------------
PP = PreProcessing()
number_cols = [f'N.{i+1}' for i in range(10)]
time_cols = ['Data', 'Ora']

# split dataset into subsets that are given as model inputs
#------------------------------------------------------------------------------
samples = df_predictions.shape[0]
extractions = df_predictions[number_cols]
times = df_predictions[time_cols]
specials = df_predictions['Numerone']

# convert time data and encode daytime
#------------------------------------------------------------------------------ 
drop_cols = ['Data', 'Ora', 'date']
times['daytime'] = pd.to_datetime(times['Ora']).dt.hour * 60 + pd.to_datetime(times['Ora']).dt.minute
times['date'] = (pd.to_datetime(times['Data']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
times = times.drop(drop_cols, axis = 1)

# generate windowed version of each input dataset
#------------------------------------------------------------------------------
X_extractions, _ = PP.timeseries_labeling(extractions, parameters['Window size']) 
X_times, _ = PP.timeseries_labeling(times, parameters['Window size'])    
X_specials, _ = PP.timeseries_labeling(specials, parameters['Window size']) 

# [PERFORM PREDICTIONS]
#==============================================================================
# ....
#==============================================================================
print('''Perform prediction using the loaded model
''')

# generate input windows for predictions
#------------------------------------------------------------------------------ 
predictions_inputs = [X_times, X_extractions, X_specials]
last_timepoints = times.tail(parameters['Window size'])
last_extractions = extractions.tail(parameters['Window size'])
last_special = specials.tail(parameters['Window size'])
last_windows = [np.reshape(last_timepoints, (1, parameters['Window size'], 1)),
                np.reshape(last_extractions, (1, parameters['Window size'], 10)),   
                np.reshape(last_special, (1, parameters['Window size'], 1))]

# perform prediction with selected model
#------------------------------------------------------------------------------
probability_vectors = model.predict(predictions_inputs)
next_prob_vectors = model.predict(last_windows)

# create list of expected extractions and their probility vector
#------------------------------------------------------------------------------
expected_extractions = []
next_expected_extractions = []
for vector in probability_vectors[0]:
    super_vector = [i+1 for i, x in enumerate(vector) if x in sorted(vector, reverse=True)[:10]]    
    expected_extractions.append(super_vector)                      
expected_special = np.argmax(probability_vectors[1], axis=-1).reshape(-1, 1) + 1

for vector in next_prob_vectors[0]:
    super_vector = [i+1 for i, x in enumerate(vector) if x in sorted(vector, reverse=True)[:10]]    
    next_expected_extractions.append(super_vector)                      
next_expected_special = np.argmax(next_prob_vectors[1], axis=-1).reshape(-1, 1) + 1

# synchronize the window of timesteps with the predictions
#------------------------------------------------------------------------------ 
sync_ext_vectors = {f'Probability of {i+1} in extraction' : [] for i in range(20)}
sync_ext_values = {f'Predicted N.{i+1}' : [] for i in range(10)}
sync_sp_vectors = {f'Probability of {i+1} as special' : [] for i in range(20)}
sync_sp_values = []

for ts in range(parameters['Window size']):
    for N in sync_ext_vectors.keys():
        sync_ext_vectors[N].append('')
    for N in sync_ext_values.keys(): 
        sync_ext_values[N].append('')
    for N in sync_sp_vectors.keys(): 
        sync_sp_vectors[N].append('')
    sync_sp_values.append('') 

for x, y, s, z in zip(probability_vectors[0], probability_vectors[1], expected_extractions, expected_special):
    for i, N in enumerate(sync_ext_vectors.keys()):
        sync_ext_vectors[N].append(x[i])
    for i, N in enumerate(sync_ext_values.keys()): 
        sync_ext_values[N].append(s[i]) 
    for i, N in enumerate(sync_sp_vectors.keys()): 
        sync_sp_vectors[N].append(y[i])    
    sync_sp_values.append(z) 

for x, y, s, z in zip(next_prob_vectors[0], next_prob_vectors[1], next_expected_extractions, next_expected_special):
    for i, N in enumerate(sync_ext_vectors.keys()):
        sync_ext_vectors[N].append(x[i])
    for i, N in enumerate(sync_ext_values.keys()): 
        sync_ext_values[N].append(s[i]) 
    for i, N in enumerate(sync_sp_vectors.keys()): 
        sync_sp_vectors[N].append(y[i])    
    sync_sp_values.append(z) 
    

# add column with prediction to dataset
#------------------------------------------------------------------------------
    timeseries.loc[len(timeseries.index)] = None
    timeseries['extraction'] = original_names
    timeseries['predicted extraction'] = sync_expected_color    
    df_probability = pd.DataFrame(sync_expected_vector)
    df_merged = pd.concat([timeseries, df_probability], axis=1)  

# add column with prediction to a new dataset and merge it with the input predictions
#------------------------------------------------------------------------------
df_forecast = pd.DataFrame()
for key, item in sync_ext_values.items():
    df_forecast[key] = item
df_forecast['predicted special'] = sync_sp_values    
for key, item in sync_ext_vectors.items():
    df_forecast[key] = item
for key, item in sync_sp_vectors.items():
    df_forecast[key] = item

df_merge = pd.concat([df_predictions, df_forecast], axis=1)

# print console report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------
Next predicted number series: {expected_extractions[-1]}
Next predicted special number: {expected_special[-1]}
-------------------------------------------------------------------------------
''')
print('Probability vector from softmax (%):')
for i, (x, y) in enumerate(sync_ext_vectors.items()):    
    print(f'{x} = {round((y[-1] * 100), 3)}')

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''
-------------------------------------------------------------------------------
Saving WFL prediction file
-------------------------------------------------------------------------------
''')

file_loc = os.path.join(GlobVar.fc_path, 'WFL_predictions.csv')         
df_merge.to_csv(file_loc, index=False, sep = ';', encoding = 'utf-8')





