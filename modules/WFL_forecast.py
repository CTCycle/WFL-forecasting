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

# Load model and associated parameters and show the model structure summary
#------------------------------------------------------------------------------
trainworker = ModelTraining(device = cnf.training_device) 
model = trainworker.load_pretrained_model(GlobVar.model_path)
parameters = trainworker.model_configuration
model.summary(expand_nested=True)

# Load scikit-learn normalizer and one-hot encoders
#------------------------------------------------------------------------------
encoder_path = os.path.join(GlobVar.pp_path, 'OHE_encoder_extractions.pkl')
with open(encoder_path, 'rb') as file:
    encoder_ext = pickle.load(file)
encoder_path = os.path.join(GlobVar.pp_path, 'OHE_encoder_special.pkl')
with open(encoder_path, 'rb') as file:
    encoder_sp = pickle.load(file)
encoder_path = os.path.join(GlobVar.pp_path, 'normalizer.pkl')
with open(encoder_path, 'rb') as file:
    normalizer = pickle.load(file)

# [PREPROCESS DATA]
#==============================================================================
# ...
#==============================================================================

# define subsets of columns to split datasets
#------------------------------------------------------------------------------
PP = PreProcessing()
number_cols = [f'N.{i+1}' for i in range(10)]
stats_cols = ['sum extractions', 'mean extractions']
info_cols = ['Concorso', 'Data', 'Ora']

# split dataset into subsets that are given as model inputs
#------------------------------------------------------------------------------
samples = df_predictions.shape[0]
extractions = df_predictions[number_cols]
times = df_predictions[info_cols]
statistics = df_predictions[stats_cols]
specials = df_predictions['Numerone']

# normalize stats features (sum and mean of extractions)
#------------------------------------------------------------------------------ 
statistics = normalizer.transform(statistics)

# generate windowed version of each input dataset
#------------------------------------------------------------------------------
X_extractions, _ = PP.timeseries_labeling(extractions, parameters['Window size']) 
X_times, _ = PP.timeseries_labeling(times, parameters['Window size'])    
X_statistics, _ = PP.timeseries_labeling(statistics, parameters['Window size'])  
X_specials, _ = PP.timeseries_labeling(specials, parameters['Window size'])     

# determine categories that are considered as possible (for each extraction position)
#------------------------------------------------------------------------------
categories = []
for i, col in enumerate(extractions):    
    real_domain = [x for x in range(i+1, i+12)]
    categories.append(real_domain)

# [PERFORM PREDICTIONS]
#==============================================================================
# ....
#==============================================================================
print('''Perform prediction using the loaded model
''')

# predict using pretrained model
#------------------------------------------------------------------------------ 
predictions_inputs = [X_extractions, X_specials]
probability_vectors = model.predict(predictions_inputs)


expected_class = np.argmax(probability_vectors, axis=-1)
last_window = CCM_timeseries.to_list()[-parameters['Window size']:]
last_window = np.reshape(last_window, (1, parameters['Window size'], 1))
next_prob_vector = model.predict(last_window)
next_exp_class = np.argmax(next_prob_vector, axis=-1)

# inverse encoding of the classes
#------------------------------------------------------------------------------ 
expected_class = np.array(expected_class).reshape(-1, 1)
next_exp_class = np.array(next_exp_class).reshape(-1, 1)
original_class = np.array(CCM_timeseries.to_list()).reshape(-1, 1)    
if parameters['Class encoding'] == True:   
    expected_color = encoder.inverse_transform(expected_class)       
    next_exp_color = encoder.inverse_transform(next_exp_class)
    original_names = encoder.inverse_transform(original_class)     
    expected_color = expected_color.flatten().tolist() 
    next_exp_color = next_exp_color.flatten().tolist()[0]   
    original_names = np.append(original_names.flatten().tolist(), '?')
    sync_expected_vector = {'Green' : [], 'Black' : [], 'Red' : []}
else:
    expected_color = expected_class.flatten().tolist() 
    next_exp_color = next_exp_class.flatten().tolist()[0]   
    original_names = np.append(original_class.flatten().tolist(), '?')
    sync_expected_vector = {f'{i}': [] for i in range(37)}

# synchronize the window of timesteps with the predictions
#------------------------------------------------------------------------------ 
sync_expected_color = []
for ts in range(cnf.window_size):
    if parameters['Class encoding'] == True:  
        sync_expected_vector['Green'].append('')
        sync_expected_vector['Black'].append('')
        sync_expected_vector['Red'].append('')
        sync_expected_color.append('')
    else:
        sync_expected_color.append('')
        for i in range(37):
            sync_expected_vector[f'{i}'].append('')            
        
for x, z in zip(probability_vectors, expected_color):   
    if parameters['Class encoding'] == True:  
        sync_expected_vector['Green'].append(x[0,0])
        sync_expected_vector['Black'].append(x[0,1])
        sync_expected_vector['Red'].append(x[0,2])
        sync_expected_color.append(z)
    else:
        sync_expected_color.append(z)
        for i in range(37):
            sync_expected_vector[f'{i}'].append(x[0,i])            

for i in range(next_prob_vector.shape[1]):
    if parameters['Class encoding'] == True:  
        sync_expected_vector['Green'].append(next_prob_vector[0,i,0])
        sync_expected_vector['Black'].append(next_prob_vector[0,i,1])
        sync_expected_vector['Red'].append(next_prob_vector[0,i,2])
        sync_expected_color.append(next_exp_color)
    else:
        sync_expected_color.append(next_exp_color)
        for r in range(37):
            sync_expected_vector[f'{r}'].append(next_prob_vector[0,i,r])

# add column with prediction to dataset
#------------------------------------------------------------------------------
CCM_timeseries.loc[len(CCM_timeseries.index)] = None
CCM_timeseries['expected color'] = sync_expected_color
CCM_timeseries['color encoding'] = original_names
df_probability = pd.DataFrame(sync_expected_vector)
df_merged = pd.concat([CCM_timeseries, df_probability], axis=1)

# print console report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------
Next predicted color: {next_exp_color}
-------------------------------------------------------------------------------
''')
print('Probability vector from softmax (%):')
for i, (x, y) in enumerate(sync_expected_vector.items()):
    print(f'{x} = {round((next_prob_vector[0,0,i] * 100), 4)}')

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''Saving CCM_predictions file (as CSV)
''')
file_loc = os.path.join(GlobVar.pp_path, 'CCM_predictions.csv')         
df_merged.to_csv(file_loc, index=False, sep = ';', encoding = 'utf-8')





