import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Embedding, Reshape, Input, Concatenate, MultiHeadAttention
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
      
    def __init__(self, plot_path, validation=True):        
        super().__init__()
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.metric_hist = []
        self.loss_val_hist = []        
        self.metric_val_hist = []
        self.validation = validation
        self.loss2_hist = []
        self.metric2_hist = []
        self.loss2_val_hist = []        
        self.metric2_val_hist = []            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):       
        if epoch % 10 == 0:                    
            self.epochs.append(epoch)
            self.loss_hist.append(logs[list(logs.keys())[1]])
            self.loss2_hist.append(logs[list(logs.keys())[2]])
            self.metric_hist.append(logs[list(logs.keys())[3]])
            self.metric2_hist.append(logs[list(logs.keys())[4]])
            if self.validation==True:
                self.loss_val_hist.append(logs[list(logs.keys())[6]])  
                self.loss2_val_hist.append(logs[list(logs.keys())[7]])          
                self.metric_val_hist.append(logs[list(logs.keys())[8]])                       
                self.metric2_val_hist.append(logs[list(logs.keys())[9]])
        if epoch % 10 == 0:            
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 2, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss, sequence')
            plt.plot(self.epochs, self.loss2_hist, label = 'training loss, special')
            if self.validation:
                plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss, sequence')
                plt.plot(self.epochs, self.loss2_val_hist, label = 'validation loss, special')
            plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot')
            plt.ylabel('Loss')
            plt.xlabel('epoch')
            plt.subplot(2, 2, 2)
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics, sequence') 
            plt.plot(self.epochs, self.metric2_hist, label = 'train metrics, special') 
            if self.validation: 
                plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics, sequence') 
                plt.plot(self.epochs, self.metric2_val_hist, label = 'validation metrics, special') 
            plt.legend(loc = 'best', fontsize = 8)
            plt.title('Metrics plot')
            plt.ylabel('Metrics')
            plt.xlabel('epoch')
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            plt.close()
                  

# Class for preprocessing tabular data prior to GAN training 
#==============================================================================
#==============================================================================
#==============================================================================
class MultiSeqWFL:

    def __init__(self, learning_rate, window_size, time_embedding, embedding_seq, 
                 embedding_special, kernel_size, seed=42, XLA_state=False):

        self.num_features = 10
        self.num_stats = 2
        self.num_classes_ext = 11
        self.num_classes_special = 20
        self.learning_rate = learning_rate        
        self.window_size = window_size
        self.time_embedding = time_embedding
        self.seq_embedding = embedding_seq
        self.special_embedding = embedding_special
        self.kernel_size = kernel_size                       
        self.seed = seed       
        self.XLA_state = XLA_state 

    #--------------------------------------------------------------------------
    def _time_encoder(self):        
        
        sequence_input = Input(shape=(self.window_size, 1))        
        embedding = Embedding(input_dim=18, output_dim=self.time_embedding)(sequence_input)               
        reshape = Reshape((self.window_size, -1))(embedding)
        #----------------------------------------------------------------------                         
        lstm1 = LSTM(32, use_bias=True, return_sequences=True, activation='tanh', 
                     kernel_initializer='glorot_uniform', dropout=0.2)(reshape)                
        lstm2 = LSTM(64, use_bias=True, return_sequences=False, activation='tanh', 
                     kernel_initializer='glorot_uniform', dropout=0.2)(lstm1)        
        bnorm = BatchNormalization()(lstm2)
        #----------------------------------------------------------------------
        output = Dense(64, kernel_initializer='he_uniform', activation='relu')(bnorm)
        #----------------------------------------------------------------------
        self.time_encoder = Model(inputs=sequence_input, outputs=output)

        return self.time_encoder

    #--------------------------------------------------------------------------
    def _sequence_encoder(self):        
        
        sequence_input = Input(shape=(self.window_size, self.num_features))        
        embedding = Embedding(input_dim=self.num_classes_ext, 
                              output_dim=self.seq_embedding)(sequence_input)
        #----------------------------------------------------------------------
        conv1 = Conv2D(64, kernel_size=self.kernel_size, kernel_initializer='he_uniform', 
                       padding='same', activation='relu')(embedding)  
        maxpool1 = MaxPooling2D()(conv1)              
        conv2 = Conv2D(128, kernel_size=self.kernel_size, kernel_initializer='he_uniform', 
                       padding='same', activation='relu')(maxpool1)  
        maxpool2 = MaxPooling2D()(conv2)           
        #----------------------------------------------------------------------                         
        reshape = Reshape((maxpool2.shape[1], maxpool2.shape[2] * maxpool2.shape[3]))(maxpool2)    
        lstm1 = LSTM(256, use_bias=True, return_sequences=True, kernel_initializer='glorot_uniform',
                     activation='tanh', dropout=0.2)(reshape)                
        lstm2 = LSTM(256, use_bias=True, return_sequences=False, kernel_initializer='glorot_uniform',
                     activation='tanh', dropout=0.2)(lstm1)        
        bnorm = BatchNormalization()(lstm2)
        #----------------------------------------------------------------------
        output = Dense(128, kernel_initializer='he_uniform', activation='relu')(bnorm)
        #----------------------------------------------------------------------
        self.sequence_encoder = Model(inputs=sequence_input, outputs=output)

        return self.sequence_encoder
    
    #--------------------------------------------------------------------------
    def _specialnum_encoder(self):  

        special_input = Input(shape=(self.window_size, 1))         
        embedding = Embedding(input_dim=self.num_classes_special, output_dim=self.special_embedding)(special_input)  
        reshape = Reshape((self.window_size, -1))(embedding)       
        #----------------------------------------------------------------------         
        lstm1 = LSTM(64, use_bias=True, return_sequences=True, kernel_initializer='glorot_uniform',
                     activation='tanh', dropout=0.2)(reshape)                
        lstm2 = LSTM(64, use_bias=True, return_sequences=False, kernel_initializer='glorot_uniform',
                     activation='tanh', dropout=0.2)(lstm1)        
        bnorm = BatchNormalization()(lstm2)
        #----------------------------------------------------------------------
        output = Dense(32, kernel_initializer='he_uniform', activation='relu')(bnorm)
        #----------------------------------------------------------------------
        self.specialnum_encoder = Model(inputs=special_input, outputs=output)

        return self.specialnum_encoder

    #--------------------------------------------------------------------------
    def _concatenated_encoder(self):        

        time_input = Input(shape=(self.time_encoder.output_shape[1:]))
        sequence_input = Input(shape=(self.sequence_encoder.output_shape[1:]))
        special_input = Input(shape=(self.specialnum_encoder.output_shape[1:]))
        #----------------------------------------------------------------------
        concat = Concatenate()([time_input, sequence_input, special_input])        
        bnorm1 = BatchNormalization()(concat)
        densecat1 = Dense(256, kernel_initializer='he_uniform', activation='relu')(bnorm1)
        bnorm2 = BatchNormalization()(densecat1) 
        densecat2 = Dense(128, kernel_initializer='he_uniform', activation='relu')(bnorm2)
        bnorm3 = BatchNormalization()(densecat2) 
        #----------------------------------------------------------------------           
        output = Dense(96, kernel_initializer='he_uniform', activation='relu')(bnorm3)
        #----------------------------------------------------------------------    
        self.concatenated_encoder = Model(inputs=[time_input, sequence_input, special_input], outputs=output)                  

        return self.concatenated_encoder

    #--------------------------------------------------------------------------
    def _sequence_decoder(self):       
        
        sequence_input = Input(shape=(self.sequence_encoder.output_shape[1:]))
        concat_input = Input(shape=(self.concatenated_encoder.output_shape[1:]))
        #----------------------------------------------------------------------
        concat = Concatenate()([sequence_input, concat_input])           
        dense1 = Dense(256, kernel_initializer='he_uniform', activation='relu')(concat)   
        bnorm = BatchNormalization()(dense1)              
        dense2 = Dense(128, kernel_initializer='he_uniform', activation='relu')(bnorm)  
        bnorm2 = BatchNormalization()(dense2)              
        dense3 = Dense(64, kernel_initializer='he_uniform', activation='relu')(bnorm2) 
        #----------------------------------------------------------------------       
        output = Dense(20, activation='sigmoid', dtype='float32')(dense3)        
        #----------------------------------------------------------------------
        self.sequence_decoder = Model(inputs=[sequence_input, concat_input], outputs=output,
                                      name='sequence_decoder')

        return self.sequence_decoder

    #--------------------------------------------------------------------------
    def _specialnum_decoder(self):        
        
        special_input = Input(shape=(self.specialnum_encoder.output_shape[1:]))
        concat_input = Input(shape=(self.concatenated_encoder.output_shape[1:]))        
        #----------------------------------------------------------------------
        concat = Concatenate()([special_input, concat_input])          
        dense = Dense(64, kernel_initializer='he_uniform', activation='relu')(concat)
        bnorm = BatchNormalization()(dense)   
        #----------------------------------------------------------------------                 
        output = Dense(self.num_classes_special, activation='softmax', dtype='float32')(bnorm)
        #----------------------------------------------------------------------
        self.specialnum_decoder = Model(inputs=[special_input, concat_input], 
                                        outputs=output, name='special_decoder')

        return self.specialnum_decoder


    #--------------------------------------------------------------------------
    def build(self):

        time_encoder = self._time_encoder()
        sequence_encoder = self._sequence_encoder()
        special_encoder = self._specialnum_encoder()
        concat_encoder = self._concatenated_encoder()      
        sequence_decoder = self._sequence_decoder() 
        special_decoder = self._specialnum_decoder() 
        #----------------------------------------------------------------------
        time_input = Input(shape=(self.window_size, 1))   
        sequence_input = Input(shape=(self.window_size, self.num_features))   
        special_input = Input(shape=(self.window_size, 1))        
        #----------------------------------------------------------------------
        time_head = time_encoder(time_input)
        sequence_head = sequence_encoder(sequence_input)
        special_head = special_encoder(special_input)        
        concatenation = concat_encoder([time_head, sequence_head, special_head]) 
        #----------------------------------------------------------------------
        sequence_output = sequence_decoder([sequence_head, concatenation])
        special_output = special_decoder([special_head, concatenation])
        #----------------------------------------------------------------------        
        inputs = [time_input, sequence_input, special_input]
        outputs = [sequence_output, special_output]
        model = Model(inputs = inputs, outputs = outputs, name = 'MultiSeqWFL_model')    
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_ext = keras.losses.BinaryCrossentropy()
        loss_special = keras.losses.CategoricalCrossentropy(from_logits=False)
        metric_ext = keras.metrics.AUC(multi_label=True)
        metric_special = keras.metrics.CategoricalAccuracy()
        model.compile(optimizer = opt, loss = [loss_ext, loss_special],                       
                      metrics={'sequence_decoder': metric_ext, 
                               'special_decoder': metric_special},
                      jit_compile=self.XLA_state)       
        
        return model             


# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class ModelTraining:    
       
    def __init__(self, device = 'default', seed=42, use_mixed_precision=False):                     
        np.random.seed(seed)
        tf.random.set_seed(seed)         
        self.available_devices = tf.config.list_physical_devices()
        print('-------------------------------------------------------------------------------')        
        print('The current devices are available: ')
        print('-------------------------------------------------------------------------------')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('-------------------------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            if not self.physical_devices:
                print('No GPU found. Falling back to CPU')
                tf.config.set_visible_devices([], 'GPU')
            else:
                if use_mixed_precision == True:
                    policy = keras.mixed_precision.Policy('mixed_float16')
                    keras.mixed_precision.set_global_policy(policy) 
                tf.config.set_visible_devices(self.physical_devices[0], 'GPU')                 
                print('GPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('-------------------------------------------------------------------------------')
            print()
    
    #========================================================================== 
    def model_parameters(self, parameters_dict, savepath):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            savepath (str): The directory path where the parameters will be saved.

        Returns:
            None       

        '''
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f) 
          
    # sequential model as generator with Keras module
    #========================================================================== 
    def load_pretrained_model(self, path, load_parameters=True):

        '''
        Load pretrained keras model (in folders) from the specified directory. 
        If multiple model directories are found, the user is prompted to select one,
        while if only one model directory is found, that model is loaded directly.
        If `load_parameters` is True, the function also loads the model parameters 
        from the target .json file in the same directory. 

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                              model parameters from a JSON file. 
                                              Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.

        '''        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Please select a pretrained model:') 
            print()
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')        
            print()               
            while True:
                try:
                    dir_index = int(input('Select the pretrained model: '))
                    print()
                except:
                    continue
                break                         
            while dir_index not in index_list:
                try:
                    dir_index = int(input('No valid model selected! Try again: '))
                    print()
                except:
                    continue
            self.model_path = os.path.join(path, model_folders[dir_index - 1])
        elif len(model_folders) == 1:
            self.model_path = os.path.join(path, model_folders[0])            
        
        model = keras.models.load_model(self.model_path)
        if load_parameters==True:
            path = os.path.join(self.model_path, 'model_parameters.json')
            with open(path, 'r') as f:
                self.model_configuration = json.load(f)            
        
        return model   


# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class ModelValidation:

    def __init__(self, model):      
        self.model = model       
    
    # comparison of data distribution using statistical methods 
    #==========================================================================     
    def WFL_confusion(self, Y_real, predictions, name, path, dpi=400):         
        cm = confusion_matrix(Y_real, predictions)    
        fig, ax = plt.subplots()        
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)        
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.set_xticks(np.arange(len(np.unique(Y_real))))
        ax.set_yticks(np.arange(len(np.unique(predictions))))
        ax.set_xticklabels(np.unique(Y_real))
        ax.set_yticklabels(np.unique(predictions))
        plt.tight_layout()
        plot_loc = os.path.join(path, f'confusion_matrix_{name}.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi = dpi)

    # comparison of data distribution using statistical methods 
    #==========================================================================
    def plot_multi_ROC(Y_real, predictions, class_dict, path, dpi):
    
        Y_real_bin = label_binarize(Y_real, classes=list(class_dict.values()))
        n_classes = Y_real_bin.shape[1]        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_real_bin[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])    
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(list(class_dict.keys())[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")       
        plot_loc = os.path.join(path, 'multi_ROC.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=dpi)           
    

