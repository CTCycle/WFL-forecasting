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
stats_cols = ['sum extractions', 'mean extractions']
time_cols = ['Data', 'Ora']

# split dataset into subsets that are given as model inputs
#------------------------------------------------------------------------------
samples = df_predictions.shape[0]
extractions = df_predictions[number_cols]
times = df_predictions[time_cols]
statistics = df_predictions[stats_cols]
specials = df_predictions['Numerone']

# convert time data and encode daytime
#------------------------------------------------------------------------------ 
drop_cols = ['Data', 'Ora']
times['daytime'] = pd.to_datetime(times['Ora']).dt.hour * 60 + pd.to_datetime(times['Ora']).dt.minute
times['date'] = (pd.to_datetime(times['Data']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
times = times.drop(drop_cols, axis = 1)

# generate windowed version of each input dataset
#------------------------------------------------------------------------------
X_extractions, _ = PP.timeseries_labeling(extractions, parameters['Window size']) 
X_times, _ = PP.timeseries_labeling(times, parameters['Window size'])    
X_statistics, _ = PP.timeseries_labeling(statistics, parameters['Window size'])  
X_specials, _ = PP.timeseries_labeling(specials, parameters['Window size'])     


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
expected_extractions = [[1 if x > 0.5 else 0 for x in sb] for sb in probability_vectors[0]]
expected_special = np.argmax(probability_vectors[1], axis=-1).reshape(-1, 1)

# synchronize the window of timesteps with the predictions
#------------------------------------------------------------------------------ 
sync_expected_extraction = {f'N.{i+1}' : [] for i in range(10)}
sync_expected_special = []

for ts in range(parameters['Window size']):
    for N in sync_expected_extraction.keys():
        sync_expected_extraction[N].append('')       
    sync_expected_special.append('') 

for x, y in zip(probability_vectors[0], probability_vectors[1]):
    for N in list(sync_expected_extraction.keys())[:-1]:
        sync_expected_extraction[N].append(x)    
    sync_expected_special.append(y)     


# add column with prediction to dataset
#------------------------------------------------------------------------------

# for key, item in sync_expected_extraction.items():
#     df_predictions[f'predicted {key}'] = sync_expected_extraction[key]
# df_predictions['predicted special'] = sync_expected_special
# for key, vector in sync_expected_vector.items():      
#     for vec in vector:
#         if len(vec) > 0:
#             for i in range(vec.shape[0]):            
#                 column_name = f'Probability of "{i+1}" at {key}'
#                 df_vectors[column_name] = item[i]
            

# print console report
#------------------------------------------------------------------------------ 
print(f'''
-------------------------------------------------------------------------------
Next predicted number series: {expected_extractions[-1]}
Next predicted special number: {expected_special[-1]}
-------------------------------------------------------------------------------
''')
print('Probability vector from softmax (%):')
for i, (x, y) in enumerate(sync_expected_extraction.items()):
    pass
    #print(f'{x} = {round((next_prob_vector[0,0,i] * 100), 4)}')

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''Saving WFL prediction file
''')

file_loc = os.path.join(GlobVar.fc_path, 'WFL_predictions.xlsx')  
writer = pd.ExcelWriter(file_loc, engine='xlsxwriter')
df_predictions.to_excel(writer, sheet_name='predictions', index=False)
writer.close()






