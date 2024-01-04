import os
import sys
import pandas as pd
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
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
from modules.components.data_assets import PreProcessing
from modules.components.training_assets import WFLMultiSeq, RealTimeHistory, ModelTraining, ModelValidation
import modules.global_variables as GlobVar
import configurations as cnf

# [LOAD DATASETS]
#==============================================================================
# Load patient dataset and dictionaries from .csv files in the dataset folder.
# Also, create a clean version of the exploded dataset to work on
#==============================================================================
filepath = os.path.join(GlobVar.data_path, 'WFL_extractions.csv')                
df_WFL = pd.read_csv(filepath, sep= ';', encoding='utf-8')
num_samples = int(df_WFL.shape[0] * cnf.data_size)
df_WFL = df_WFL[(df_WFL.shape[0] - num_samples):]

print(f'''
-------------------------------------------------------------------------------
WFL Training (k-fold)
-------------------------------------------------------------------------------
Leverage large volume of Win For Life extractions (from the Xamig.com database)
and predict future extractions based on the observed timeseries 
''')

# [MAPPING AND ENCODING]
#==============================================================================
# ...
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
Data preprocessing
-------------------------------------------------------------------------------
      
STEP 1 -----> Split datasets and generate windows for model training
''')

PP = PreProcessing()
number_cols = [f'N.{i+1}' for i in range(10)]
time_cols = ['Data', 'Ora']

# split main dataset in subsets
#------------------------------------------------------------------------------
trainset = df_WFL.copy()
train_samples, test_samples = trainset.shape[0], 0
train_ext = trainset[number_cols]
train_time = trainset[time_cols]
train_special = trainset['Numerone'] 

# convert time data and encode daytime
#------------------------------------------------------------------------------ 
drop_cols = ['Data', 'Ora', 'date']
train_time['daytime'] = pd.to_datetime(train_time['Ora']).dt.hour * 60 + pd.to_datetime(train_time['Ora']).dt.minute
train_time['date'] = (pd.to_datetime(train_time['Data']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
train_time = train_time.drop(drop_cols, axis = 1)
  
# generate window datasets
#------------------------------------------------------------------------------
X_train_ext, Y_train_ext = PP.timeseries_labeling(train_ext, cnf.window_size)
X_train_special, Y_train_special = PP.timeseries_labeling(train_special, cnf.window_size)     
X_train_time, _ = PP.timeseries_labeling(train_time, cnf.window_size)   
 
# [ONE HOT ENCODE THE LABELS]
#==============================================================================
# ...
#==============================================================================
print('''STEP 2 -----> One-Hot encode timeseries labels (Y data) and normalize stats
''')

# determine categories
#------------------------------------------------------------------------------
all_categories = [x+1 for x in range(20)]
possible_categories = []
for i, col in enumerate(train_ext):    
    pos_categories = [x for x in range(i+1, i+12)]
    possible_categories.append(pos_categories)

# one hot encode the output for softmax training shape = (timesteps, features)
#------------------------------------------------------------------------------
OH_encoder_ext = MultiLabelBinarizer(classes=all_categories)
OH_encoder_special = OneHotEncoder(sparse=False, dtype='float32')
Y_train_ext_OHE = OH_encoder_ext.fit_transform(Y_train_ext)
Y_train_special_OHE = OH_encoder_special.fit_transform(Y_train_special.reshape(-1,1))

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''STEP 3 -----> Save preprocessed files
''')

# create model folder
#------------------------------------------------------------------------------
model_savepath = PP.model_savefolder(GlobVar.model_path, 'MultiSeqWFL')
pp_path = os.path.join(model_savepath, 'preprocessed data')
if not os.path.exists(pp_path):
    os.mkdir(pp_path)

# save normalizer and one-hot encoder
#------------------------------------------------------------------------------
encoder_path = os.path.join(pp_path, 'OHE_encoder_extractions.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_ext, file)
encoder_path = os.path.join(pp_path, 'OHE_encoder_special.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_special, file)

## [DEFINE AND BUILD MODEL]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''STEP 4 -----> Build the model and start training
''')
model_savepath = PP.model_savefolder(GlobVar.model_path, 'WFLMulti')
trainworker = ModelTraining(device=cnf.training_device, seed=cnf.seed, 
                            use_mixed_precision=cnf.use_mixed_precision)

# initialize model class
#------------------------------------------------------------------------------
modelframe = WFLMultiSeq(cnf.learning_rate, cnf.window_size, cnf.embedding_time, 
                         cnf.embedding_sequence, cnf.embedding_special, cnf.kernel_size, 
                         seed=cnf.seed, XLA_state=cnf.XLA_acceleration)
model = modelframe.build()
model.summary(expand_nested=True)

# plot model graph
#------------------------------------------------------------------------------
if cnf.generate_model_graph == True:
    plot_path = os.path.join(model_savepath, 'MultiSeqWFL_model.png')       
    plot_model(model, to_file = plot_path, show_shapes = True, 
               show_layer_names = True, show_layer_activations = True, 
               expand_nested = True, rankdir = 'TB', dpi = 400)

# [TRAINING WITH WFL]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir = tensorboard/
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
TRAINING INFO
-------------------------------------------------------------------------------
Number of k-fold runs:     {cnf.k_fold}
Number of epochs per run : {cnf.k_epochs}
Window size:               {cnf.window_size}
Batch size:                {cnf.batch_size} 
Learning rate:             {cnf.learning_rate} 
-------------------------------------------------------------------------------  
''')

# define k fold strategy
#------------------------------------------------------------------------------
kfold = TimeSeriesSplit(n_splits=cnf.k_fold)

# training loop with k fold and save model at the end
#------------------------------------------------------------------------------
model_scores = []
for train, test in kfold.split(X_train_ext):
    RTH_callback = RealTimeHistory(model_savepath, validation=cnf.use_test_data)
    train_model_inputs = [X_train_time[train], X_train_ext[train], X_train_special[train]]
    train_model_outputs = [Y_train_ext_OHE[train], Y_train_special_OHE[train]]    
    test_data = [[X_train_time[test], X_train_ext[test], X_train_special[test]],
                 [Y_train_ext_OHE[test], Y_train_special_OHE[test]]]    
    training = model.fit(x=train_model_inputs, y=train_model_outputs, batch_size=cnf.batch_size, 
                     validation_data=test_data, epochs = cnf.k_epochs, verbose=1, shuffle=False, 
                     callbacks = [RTH_callback], workers = 6, use_multiprocessing=True)    
    score = model.evaluate(test_data[0], verbose=0)
    model_scores.append(score)

# save model data and parameters in txt files
#------------------------------------------------------------------------------
parameters = {'Number of train samples' : train_samples,
              'Number of test samples' : test_samples,             
              'Window size' : cnf.window_size,              
              'Embedding dimensions (sequences)' : cnf.embedding_sequence,  
              'Embedding dimensions (special)' : cnf.embedding_special,            
              'Batch size' : cnf.batch_size,
              'Learning rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs}

model.save(model_savepath)
trainworker.model_parameters(parameters, model_savepath)

print(model_scores)

