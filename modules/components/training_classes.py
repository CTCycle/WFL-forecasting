import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv1D, Conv2D, MaxPooling1D, AveragePooling2D
from keras.layers import Embedding, Reshape, Input, RepeatVector, TimeDistributed, Concatenate, MultiHeadAttention
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):
    
    """ 
    A class including the callback to show a real time plot of the training history. 
      
    Methods:
        
    __init__(plot_path): initializes the class with the plot savepath       
    
    """   
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
        if epoch % 20 == 0:            
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

    def __init__(self, learning_rate, window_size, embedding_seq, embedding_special,
                 kernel_size, neurons, seed=42, XLA_state=False):

        self.num_features = 10
        self.num_stats = 2
        self.num_classes_ext = 11
        self.num_classes_special = 20
        self.learning_rate = learning_rate        
        self.window_size = window_size
        self.seq_embedding = embedding_seq
        self.special_embedding = embedding_special
        self.kernel_size = kernel_size        
        self.neurons = neurons                
        self.seed = seed       
        self.XLA_state = XLA_state        

    def build(self):
       

        #----------------------------------------------------------------------
        # SEQUENCE INPUT
        #----------------------------------------------------------------------
        sequence_input = Input(shape=(self.window_size, self.num_features))
        #----------------------------------------------------------------------
        # embedding heads
        #---------------------------------------------------------------------- 
        embedding1 = Embedding(input_dim=self.num_classes_ext, output_dim=self.seq_embedding)(sequence_input)       
        #----------------------------------------------------------------------
        # convolutional block
        #----------------------------------------------------------------------
        conv1 = Conv2D(self.seq_embedding, kernel_size=self.kernel_size, padding='same', activation='relu')(embedding1)  
        maxpool1 = AveragePooling2D()(conv1)              
        conv2 = Conv2D(self.neurons*4, kernel_size=self.kernel_size, padding='same', activation='relu')(maxpool1)  
        maxpool2 = AveragePooling2D()(conv2)              
        conv3 = Conv2D(self.neurons*2, kernel_size=self.kernel_size, padding='same', activation='relu')(maxpool2)
        
        #----------------------------------------------------------------------
        # tensor reshaping fro 4D to 3D
        #----------------------------------------------------------------------  
        conv_out_shape = conv3.shape           
        reshape = Reshape((conv_out_shape[1], conv_out_shape[2] * conv_out_shape[3]))(conv3)
        #----------------------------------------------------------------------
        # lstm block (recurrent network)
        #----------------------------------------------------------------------                 
        lstm1 = LSTM(self.neurons*4, use_bias=True, return_sequences=True, activation='tanh', dropout=0.2)(reshape)                
        lstm2 = LSTM(self.neurons*2, use_bias=True, return_sequences=False, activation='tanh', dropout=0.2)(lstm1)
        #----------------------------------------------------------------------
        # post recurrent dense block
        #----------------------------------------------------------------------
        bnormrn1 = BatchNormalization()(lstm2)
        densern1 = Dense(self.neurons*2, kernel_initializer='he_uniform', activation='relu')(bnormrn1)
        

        #----------------------------------------------------------------------
        # SPECIAL NUMBER INPUT
        #----------------------------------------------------------------------
        special_input = Input(shape=(self.window_size, 1)) 
        #----------------------------------------------------------------------
        # embedding heads
        #----------------------------------------------------------------------        
        embedding2 = Embedding(input_dim=self.num_classes_special, output_dim=self.special_embedding)(special_input)  
        reshape2 = Reshape((self.window_size, -1))(embedding2)
        #----------------------------------------------------------------------
        # convolutional block
        #----------------------------------------------------------------------         
        conv4 = Conv1D(self.special_embedding, kernel_size=self.kernel_size, padding='same', activation='relu')(reshape2)                   
        conv5 = Conv1D(self.neurons*4, kernel_size=self.kernel_size, padding='same', activation='relu')(conv4)               
        conv6 = Conv1D(self.neurons*2, kernel_size=self.kernel_size, padding='same', activation='relu')(conv5)  
                        
        #----------------------------------------------------------------------
        # lstm block (recurrent network)
        #----------------------------------------------------------------------         
        lstm3 = LSTM(self.neurons*2, use_bias=True, return_sequences=True, activation='tanh', dropout=0.2)(conv6)                
        lstm4 = LSTM(self.neurons*2, use_bias=True, return_sequences=False, activation='tanh', dropout=0.2)(lstm3)
        #----------------------------------------------------------------------
        # post recurrent dense block
        #----------------------------------------------------------------------
        bnormsp1 = BatchNormalization()(lstm4)
        densesp1 = Dense(self.neurons, kernel_initializer='he_uniform', activation='relu')(bnormsp1)
        
        
        #----------------------------------------------------------------------
        # CONCATENATED BLOCK
        #----------------------------------------------------------------------         
        concat1 = Concatenate()([densern1, densesp1])
        bnorm1 = BatchNormalization()(concat1)
        densecat1 = Dense(self.neurons*4, kernel_initializer='he_uniform', activation='relu')(bnorm1)
        bnorm2 = BatchNormalization()(densecat1) 
        densecat2 = Dense(self.neurons*2, kernel_initializer='he_uniform', activation='relu')(bnorm2)
        bnorm3 = BatchNormalization()(densecat2)            
        densecat3 = Dense(self.neurons, kernel_initializer='he_uniform', activation='relu')(bnorm3)  

        #----------------------------------------------------------------------
        # REPEATED VECTOR TAIL
        #----------------------------------------------------------------------               
        concat2 = Concatenate()([densern1, densecat3])
        repeat = RepeatVector(n=self.num_features)(concat2)
        bnorm4 = BatchNormalization()(repeat)   
        denserep1 = Dense(self.neurons*4, kernel_initializer='he_uniform', activation='relu')(bnorm4)   
        bnorm5 = BatchNormalization()(denserep1)              
        denserep2 = Dense(self.neurons*2, kernel_initializer='he_uniform', activation='relu')(bnorm5) 
        #----------------------------------------------------------------------                     
        # set outputs           
        #---------------------------------------------------------------------- 
        output_seq = TimeDistributed(Dense(self.num_classes_ext, activation='softmax', 
                                           name='extractions', dtype='float32'))(denserep2)

        #----------------------------------------------------------------------
        # DENSE TAIL FOR SPECIAL NUMBER
        #----------------------------------------------------------------------              
        concat3 = Concatenate()([densesp1, densecat3])
        bnorm6 = BatchNormalization()(concat3)  
        densesp1 = Dense(self.neurons*2, kernel_initializer='he_uniform', activation='relu')(bnorm6)
        bnorm7 = BatchNormalization()(densesp1)                
        densesp2 = Dense(self.neurons, kernel_initializer='he_uniform', activation='relu')(bnorm7)
        #----------------------------------------------------------------------                     
        # set outputs           
        #----------------------------------------------------------------------       
        output_special = Dense(self.num_classes_special, activation='softmax', 
                               name='special_number', dtype='float32')(densesp2)

        #----------------------------------------------------------------------
        # COMPILE MODEL
        #----------------------------------------------------------------------  
        inputs = [sequence_input, special_input]
        outputs = [output_seq, output_special]
        model = Model(inputs = inputs, outputs = outputs, name = 'MultiSeqWFL_model')    
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        metrics = keras.metrics.CategoricalAccuracy()
        model.compile(loss = loss, optimizer = opt, metrics = metrics,
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
        path = os.path.join(savepath, 'model_parameters.json')      
        with open(path, 'w') as f:
            json.dump(parameters_dict, f) 
          
    # sequential model as generator with Keras module
    #========================================================================== 
    def load_pretrained_model(self, path, load_parameters=True):
        
        model_folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                model_folders.append(entry.name)
        model_folders.sort()
        index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
        print('Please select a pretrained model:') 
        print()
        for i, directory in enumerate(model_folders):
            print(f'{i + 1} - {directory}')        
        print()               
        while True:
           try:
              dir_index = int(input('Type the model index to select it: '))
              print()
           except:
              continue
           break                         
        while dir_index not in index_list:
           try:
               dir_index = int(input('Input is not valid! Try again: '))
               print()
           except:
               continue  
           
        model_path = os.path.join(path, model_folders[dir_index - 1])
        model = keras.models.load_model(model_path)
        if load_parameters==True:
            path = os.path.join(model_path, 'model_parameters.json')
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
    def FAIRS_confusion(self, Y_real, predictions, name, path, dpi=400):         
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
    

