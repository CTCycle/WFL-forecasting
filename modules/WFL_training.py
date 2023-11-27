import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
from modules.components.data_classes import PreProcessing
from modules.components.training_classes import MultiSeqWFL, RealTimeHistory, ModelTraining, ModelValidation
import modules.global_variables as GlobVar
import modules.configurations as cnf

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
WFL Training
-------------------------------------------------------------------------------
Leverage large volume of Win For Life extractions (from the Xamig.com database)
and predict future extractions based on the observed timeseries 
''')

# [COLOR MAPPING AND ENCODING]
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
stats_cols = ['sum extractions', 'mean extractions']
info_cols = ['Concorso', 'Data', 'Ora']

# split dataset into train and test and split further into subsets
#------------------------------------------------------------------------------
if cnf.use_test_data == True:
    trainset, testset = PP.split_timeseries(df_WFL, cnf.test_size, inverted=cnf.invert_test)     
    train_samples, test_samples = trainset.shape[0], testset.shape[0]
    train_ext, test_ext = trainset[number_cols], testset[number_cols]
    train_time, test_time = trainset[info_cols], testset[info_cols]
    train_stats, test_stats = trainset[stats_cols], testset[stats_cols]
    train_special, test_special = trainset['Numerone'], testset['Numerone']
else:
    trainset = df_WFL.copy()
    train_samples, test_samples = trainset.shape[0], 0
    train_ext = trainset[number_cols]
    train_time = trainset[info_cols]
    train_stats = trainset[stats_cols]
    train_special = trainset['Numerone'] 

# normalize features
#------------------------------------------------------------------------------ 
normalizer = MinMaxScaler(feature_range=(0, 1))
normalizer.fit(train_stats)
train_stats = normalizer.transform(train_stats)
if cnf.use_test_data == True: 
    test_stats = normalizer.transform(test_stats)
    
# generate window datasets
#------------------------------------------------------------------------------
if cnf.use_test_data == True:    
    X_train_ext, Y_train_ext = PP.timeseries_labeling(train_ext, cnf.window_size)     
    X_test_ext, Y_test_ext = PP.timeseries_labeling(test_ext, cnf.window_size)   
    X_train_time, _ = PP.timeseries_labeling(train_time, cnf.window_size)     
    X_test_time, _ = PP.timeseries_labeling(test_time, cnf.window_size)
    X_train_stats, _ = PP.timeseries_labeling(train_stats, cnf.window_size)
    X_test_stats, _ = PP.timeseries_labeling(test_stats, cnf.window_size)
    X_train_special, Y_train_special = PP.timeseries_labeling(train_special, cnf.window_size)     
    X_test_special, Y_test_special = PP.timeseries_labeling(test_special, cnf.window_size)
else:
    X_train_ext, Y_train_ext = PP.timeseries_labeling(train_ext, cnf.window_size)    
    X_train_time, _ = PP.timeseries_labeling(train_time, cnf.window_size) 
    X_train_stats, Y_train_stats = PP.timeseries_labeling(train_stats, cnf.window_size)   
    X_train_special, Y_train_special = PP.timeseries_labeling(train_special, cnf.window_size)  

# [ONE HOT ENCODE THE LABELS]
#==============================================================================
# ...
#==============================================================================
print('''STEP 2 -----> One-Hot encode timeseries labels (Y data) and normalize stats
''')

# determine categories
#------------------------------------------------------------------------------
categories = []
for i, col in enumerate(train_ext):    
    real_domain = [x for x in range(i+1, i+12)]
    categories.append(real_domain)

# one hot encode the output for softmax training shape = (timesteps, features)
#------------------------------------------------------------------------------
OH_encoder_ext = OneHotEncoder(categories=categories, sparse=False, dtype='float32')
OH_encoder_special = OneHotEncoder(sparse=False, dtype='float32')
Y_train_ext_OHE = OH_encoder_ext.fit_transform(Y_train_ext)
Y_train_special_OHE = OH_encoder_special.fit_transform(Y_train_special.reshape(-1,1))
if cnf.use_test_data == True: 
    Y_test_ext_OHE = OH_encoder_ext.fit_transform(Y_test_ext)    
    Y_test_special_OHE = OH_encoder_special.fit_transform(Y_test_special.reshape(-1,1))

# [SAVE FILES]
#==============================================================================
# Save the trained preprocessing systems (normalizer and encoders) for further use 
#==============================================================================
print('''STEP 3 -----> Save preprocessed files
''')

# save encoder
#------------------------------------------------------------------------------
encoder_path = os.path.join(GlobVar.pp_path, 'OHE_encoder_extractions.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_ext, file)
encoder_path = os.path.join(GlobVar.pp_path, 'OHE_encoder_special.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_special, file)
encoder_path = os.path.join(GlobVar.pp_path, 'normalizer.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(normalizer, file)


# generate dataframes to savbe preprocessed data
#------------------------------------------------------------------------------
if cnf.save_files==True:
    df_X_train_ext = pd.DataFrame(X_train_ext.reshape(X_train_ext.shape[0], -1))
    df_X_train_time = pd.DataFrame(X_train_time.reshape(X_train_time.shape[0], -1))
    df_X_train_special = pd.DataFrame(X_train_special.reshape(X_train_special.shape[0], -1))
    df_Y_train_ext_OHE = pd.DataFrame(Y_train_ext_OHE)
    df_Y_train_special_OHE = pd.DataFrame(Y_train_special_OHE)
    if cnf.use_test_data == True: 
        df_Y_test_ext_OHE = pd.DataFrame(Y_test_ext_OHE)    
        df_Y_test_special_OHE = pd.DataFrame(Y_test_special_OHE)
        df_X_test_ext = pd.DataFrame(X_test_ext.reshape(X_test_ext.shape[0], -1))
        df_X_test_time = pd.DataFrame(X_test_time.reshape(X_test_time.shape[0], -1))
        df_X_test_special = pd.DataFrame(X_test_special.reshape(X_test_special.shape[0], -1))

# save excel files
#------------------------------------------------------------------------------
    file_loc = os.path.join(GlobVar.pp_path, 'WFL_preprocessed.xlsx')  
    writer = pd.ExcelWriter(file_loc, engine='xlsxwriter')
    df_X_train_ext.to_excel(writer, sheet_name='train extractions', index=False)
    df_Y_train_ext_OHE.to_excel(writer, sheet_name='train ext labels', index=False)
    df_X_train_time.to_excel(writer, sheet_name='train time', index=False)
    df_X_train_special.to_excel(writer, sheet_name='train special', index=False)
    df_Y_train_special_OHE.to_excel(writer, sheet_name='train special labels', index=False)
    if cnf.use_test_data == True:  
        df_X_test_ext.to_excel(writer, sheet_name='test extractions', index=False)
        df_Y_test_ext_OHE.to_excel(writer, sheet_name='test ext labels', index=False)
        df_X_test_time.to_excel(writer, sheet_name='test time', index=False)
        df_X_test_special.to_excel(writer, sheet_name='test special', index=False)
        df_Y_test_special_OHE.to_excel(writer, sheet_name='test special labels', index=False)

    writer.close()

# [REPORT AND ANALYSIS]
#==============================================================================
# ....
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
DATASET PROPERTIES
-------------------------------------------------------------------------------
Number of timepoints in train dataset: {train_samples}
Number of timepoints in test dataset:  {test_samples}
-------------------------------------------------------------------------------  

''')

# [DEFINE AND BUILD MODEL]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''STEP 4 -----> Build the model and start training
''')


# reshape Y data
#------------------------------------------------------------------------------
Y_train_ext_OHE = np.reshape(Y_train_ext_OHE, (Y_train_ext_OHE.shape[0], 10, 11))
if cnf.use_test_data == True: 
    Y_test_ext_OHE = np.reshape(Y_test_ext_OHE, (Y_test_ext_OHE.shape[0], 10, 11))

# activate training framework and create model folder
#------------------------------------------------------------------------------
trainworker = ModelTraining(device=cnf.training_device, seed=cnf.seed, 
                            use_mixed_precision=cnf.use_mixed_precision) 
model_savepath = PP.model_savefolder(GlobVar.model_path, 'MultiSeqWFL')

# initialize model class
#------------------------------------------------------------------------------
modelframe = MultiSeqWFL(cnf.learning_rate, cnf.window_size, cnf.embedding_sequence,
                         cnf.embedding_special, cnf.kernel_size, cnf.neuron_baseline, 
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

# [TRAINING WITH FAIRS]
#==============================================================================
# Setting callbacks and training routine for the features extraction model. 
# use command prompt on the model folder and (upon activating environment), 
# use the bash command: python -m tensorboard.main --logdir = tensorboard/
#==============================================================================
train_model_inputs = [X_train_ext, X_train_special]
train_model_outputs = [Y_train_ext_OHE, Y_train_special_OHE]
test_model_inputs = [X_test_ext, X_test_special]
test_model_outputs = [Y_test_ext_OHE, Y_test_special_OHE]

# training loop and model saving at end
#------------------------------------------------------------------------------
print(f'''Start model training for {cnf.epochs} epochs and batch size of {cnf.batch_size}
       ''')

RTH_callback = RealTimeHistory(model_savepath, validation=cnf.use_test_data)
if cnf.use_test_data == True:
    validation_data = (test_model_inputs, test_model_outputs)   
else:
    validation_data = None 

if cnf.use_tensorboard == True:
    log_path = os.path.join(model_savepath, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

training = model.fit(x=train_model_inputs, y=train_model_outputs, batch_size=cnf.batch_size, 
                     validation_data=validation_data, epochs = cnf.epochs, verbose=1, 
                     callbacks = callbacks, workers = 6, use_multiprocessing=True)

model.save(model_savepath)

# save model parameters in txt files
#------------------------------------------------------------------------------
parameters = {'Number of train samples' : train_samples,
              'Number of test samples' : test_samples,              
              'Lowest neurons number' : cnf.neuron_baseline,
              'Window size' : cnf.window_size,              
              'Embedding dimensions (sequences)' : cnf.embedding_sequence,  
              'Embedding dimensions (special)' : cnf.embedding_special,            
              'Batch size' : cnf.batch_size,
              'Learning rate' : cnf.learning_rate,
              'Epochs' : cnf.epochs}

trainworker.model_parameters(parameters, model_savepath)

# [MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
print(f'''STEP 5 -----> Evaluate the model''')

validator = ModelValidation(model)

# predict lables from train set
#------------------------------------------------------------------------------
predicted_train = model.predict(train_model_inputs, verbose=1)


# y_pred_labels = np.argmax(predicted_train, axis=-1)
# y_true_labels = np.argmax(Y_train_OHE, axis=-1)
# Y_pred, Y_true = y_pred_labels[:, 0], y_true_labels[:, 0]

# # show predicted classes (train dataset)
# #------------------------------------------------------------------------------
# class_pred, class_true = np.unique(Y_pred), np.unique(Y_true)
# print(f'''
# Number of classes observed in train (true labels): {len(class_true)}
# Number of classes observed in train (predicted labels): {len(class_pred)}
# Classes observed in predicted train labels:''')
# for x in class_pred:
#     print(x)

# # generate confusion matrix from train set (if class num is equal)
# #------------------------------------------------------------------------------
# try:
#     validator.FAIRS_confusion(Y_true, Y_pred, 'train', model_savepath)
#     #validator.FAIRS_ROC_curve(y_true_labels, y_pred_labels, categories_mapping, 'train', model_savepath, 400)
# except Exception as e:
#     print('Could not generate confusion matrix for train dataset')
#     print('Error:', str(e))

# # predict lables from test set
# #------------------------------------------------------------------------------
# if cnf.use_test_data == True:
#     predicted_test = model.predict(X_test, verbose=0)
#     y_pred_labels = np.argmax(predicted_test, axis=-1)
#     y_true_labels = np.argmax(Y_test_OHE, axis=-1)
#     Y_pred, Y_true = y_pred_labels[:, 0:1], y_true_labels[:, 0:1]

# # show predicted classes (testdataset)
# #------------------------------------------------------------------------------
#     class_pred, class_true = np.unique(Y_pred), np.unique(Y_true)
#     print(f'''
# Number of classes observed in test (true labels): {len(class_true)}
# Number of classes observed in test (predicted labels): {len(class_pred)}
# Classes observed in predicted test labels:''')
#     for x in class_pred:
#         print(x)     

# # generate confusion matrix from test set (if class num is equal)
# #------------------------------------------------------------------------------
#     try:
#         validator.FAIRS_confusion(Y_true, Y_pred, 'test', model_savepath)
#         #validator.FAIRS_ROC_curve(y_true_labels, y_pred_labels, categories_mapping, 'test', model_savepath, 400)
#     except Exception as e:
#         print('Could not generate confusion matrix for test dataset')
#         print('Error:', str(e))



