import os
import sys
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from keras.utils import plot_model

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
WFL Training
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

# split dataset into train and test, then split further into different input
# datasets (daytime, 10-point sequence timeseries, special number timeseries)
#------------------------------------------------------------------------------
if cnf.use_test_data == True:
    trainset, testset = PP.split_timeseries(df_WFL, cnf.test_size, inverted=cnf.invert_test)     
    train_samples, test_samples = trainset.shape[0], testset.shape[0]
    train_ext, test_ext = trainset[number_cols], testset[number_cols]
    train_time, test_time = trainset[time_cols], testset[time_cols]    
    train_special, test_special = trainset['Numerone'], testset['Numerone']
else:
    trainset = df_WFL.copy()
    train_samples, test_samples = trainset.shape[0], 0
    train_ext = trainset[number_cols]
    train_time = trainset[time_cols]   
    train_special = trainset['Numerone'] 

# convert date and daytime in a different format (days from 1970 for the date, 
# minutes from midnight for the daytime), keeps only the daytime as input information
#------------------------------------------------------------------------------ 
drop_cols = ['Data', 'Ora', 'date']
train_time['daytime'] = pd.to_datetime(train_time['Ora']).dt.hour * 60 + pd.to_datetime(train_time['Ora']).dt.minute
train_time['date'] = (pd.to_datetime(train_time['Data']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
train_time = train_time.drop(drop_cols, axis = 1)
if cnf.use_test_data == True:
    test_time['daytime'] = pd.to_datetime(test_time['Ora']).dt.hour * 60 + pd.to_datetime(test_time['Ora']).dt.minute
    test_time['date'] = (pd.to_datetime(test_time['Data']) - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')
    test_time = test_time.drop(drop_cols, axis = 1)

   
# generate X-Y pairs using rolling window method 
#------------------------------------------------------------------------------
X_train_ext, Y_train_ext = PP.timeseries_labeling(train_ext, cnf.window_size)
X_train_special, Y_train_special = PP.timeseries_labeling(train_special, cnf.window_size)     
X_train_time, _ = PP.timeseries_labeling(train_time, cnf.window_size) 
if cnf.use_test_data == True:           
    X_test_ext, Y_test_ext = PP.timeseries_labeling(test_ext, cnf.window_size) 
    X_test_special, Y_test_special = PP.timeseries_labeling(test_special, cnf.window_size)    
    X_test_time, _ = PP.timeseries_labeling(test_time, cnf.window_size)    
      
 
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

# one hot encode the special number labels for the special number decoder, 
# which uses softmax activation (shape of timesteps, features)
#------------------------------------------------------------------------------
OH_encoder_ext = MultiLabelBinarizer(classes=all_categories)
OH_encoder_special = OneHotEncoder(sparse=False, dtype='float32')
Y_train_ext_OHE = OH_encoder_ext.fit_transform(Y_train_ext)
Y_train_special_OHE = OH_encoder_special.fit_transform(Y_train_special.reshape(-1,1))
if cnf.use_test_data == True: 
    Y_test_ext_OHE = OH_encoder_ext.transform(Y_test_ext)    
    Y_test_special_OHE = OH_encoder_special.transform(Y_test_special.reshape(-1,1))

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

# save the one-hot encoder and the multilabel binarizer
#------------------------------------------------------------------------------
encoder_path = os.path.join(pp_path, 'MLB_encoder_extractions.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_ext, file)
encoder_path = os.path.join(pp_path, 'OHE_encoder_special.pkl')
with open(encoder_path, 'wb') as file:
    pickle.dump(OH_encoder_special, file)

# generate dataframes to save preprocessed data
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
    file_loc = os.path.join(pp_path, 'WFL_preprocessed.xlsx')  
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

# [DEFINE AND BUILD MODEL]
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
DATASET PROPERTIES
-------------------------------------------------------------------------------
Number of timepoints in train dataset: {train_samples}
Number of timepoints in test dataset:  {test_samples}
-------------------------------------------------------------------------------
TRAINING INFO
-------------------------------------------------------------------------------
Number of epochs: {cnf.epochs}
Window size:      {cnf.window_size}
Batch size:       {cnf.batch_size} 
Learning rate:    {cnf.learning_rate} 
-------------------------------------------------------------------------------  
''')

# initialize real time plot callback
#------------------------------------------------------------------------------
RTH_callback = RealTimeHistory(model_savepath, validation=cnf.use_test_data)

# create list of inputs
#------------------------------------------------------------------------------
train_model_inputs = [X_train_time, X_train_ext, X_train_special]
train_model_outputs = [Y_train_ext_OHE, Y_train_special_OHE]
if cnf.use_test_data == True:
    test_model_inputs = [X_test_time, X_test_ext, X_test_special]
    test_model_outputs = [Y_test_ext_OHE, Y_test_special_OHE]
    validation_data = (test_model_inputs, test_model_outputs)   
else:
    validation_data = None 

# initialize tensorboard
#------------------------------------------------------------------------------
if cnf.use_tensorboard == True:
    log_path = os.path.join(model_savepath, 'tensorboard')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
    callbacks = [RTH_callback, tensorboard_callback]    
else:    
    callbacks = [RTH_callback]

# training loop and save model at the end
#------------------------------------------------------------------------------
training = model.fit(x=train_model_inputs, y=train_model_outputs, batch_size=cnf.batch_size, 
                     validation_data=validation_data, epochs = cnf.epochs, verbose=1, shuffle=False, 
                     callbacks = callbacks, workers = 6, use_multiprocessing=True)

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

# [MODEL VALIDATION]
#==============================================================================
# ...
#==============================================================================
print(f'''STEP 5 -----> Evaluate the model''')

validator = ModelValidation(model)

# predict lables from train set
#------------------------------------------------------------------------------
#predicted_train = model.predict(train_model_inputs, verbose=1)
# predicted_extractions = []
# predicted_special = np.argmax(predicted_train[1], axis=-1)
# y_true_labels = np.argmax(Y_train_OHE, axis=-1)
# Y_pred, Y_true = y_pred_labels[:, 0], y_true_labels[:, 0]


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


# # generate confusion matrix from test set (if class num is equal)
# #------------------------------------------------------------------------------
#     try:
#         validator.FAIRS_confusion(Y_true, Y_pred, 'test', model_savepath)
#         #validator.FAIRS_ROC_curve(y_true_labels, y_pred_labels, categories_mapping, 'test', model_savepath, 400)
#     except Exception as e:
#         print('Could not generate confusion matrix for test dataset')
#         print('Error:', str(e))



