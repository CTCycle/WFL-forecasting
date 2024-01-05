import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Embedding, Reshape, Input, Concatenate, Add
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# [CALLBACK FOR REAL TIME TRAINING MONITORING]
#==============================================================================
# Real time monitoring callback
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
                  
# [CUSTOM LOSS]
#==============================================================================
# Multiclass model for training
#==============================================================================
class WFLossFunction(tf.keras.losses.Loss):

    '''
    Custom loss function for a TensorFlow/Keras model with additional mean difference penalty.
    This loss function combines a binary cross-entropy term with a penalty term based on the 
    absolute difference between the mean of predicted values and the mean of true values.

    Keyword Arguments:
        epsilon (float, optional): A hyperparameter controlling the strength of 
                                   the mean difference penalty. Defaults to 0.1.    
    '''

    def __init__(self, epsilon=1.0):        
        super().__init__()
        self.binarycross = keras.losses.BinaryCrossentropy()
        self.epsilon = epsilon

    # implement loss through call method  
    #--------------------------------------------------------------------------
    def call(self, y_true, y_pred): 

        '''
        Computes the total loss using binary cross-entropy and mean difference penalty.

        Parameters:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Predicted labels.

        Returns:
            tf.Tensor: Total loss.

        '''
        bc_loss = self.binarycross(y_true, y_pred)
        y_true = tf.cast(y_true, dtype=tf.float32)     
        mean_diff_loss = tf.abs(tf.reduce_mean(tf.round(y_pred)) - tf.reduce_mean(y_true))
        total_loss = bc_loss + (mean_diff_loss * self.epsilon)
        
        return total_loss

# [TIME ENCODER]
#==============================================================================
# Custom time encoder
#==============================================================================
class TimeEncoder(keras.layers.Layer):
    def __init__(self, window_size, embedding_dims):
        super(TimeEncoder, self).__init__()
        self.window_size = window_size
        self.embedding_dims = embedding_dims
        self.embedding = Embedding(input_dim=18, output_dim=self.embedding_dims)
        self.reshape = Reshape((self.window_size, -1))
        self.lstm1 = LSTM(64, use_bias=True, return_sequences=True, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)
        self.lstm2 = LSTM(128, use_bias=True, return_sequences=False, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)        
        self.dense = Dense(128, kernel_initializer='he_uniform', activation='relu')

    # implement time encoding through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs):
        layer = self.embedding(inputs)
        layer = self.reshape(layer)
        layer = self.lstm1(layer)
        layer = self.lstm2(layer)        
        output_layer = self.dense(layer)

        return output_layer

    # get configuration to print out structure
    #--------------------------------------------------------------------------
    def get_config(self):
        config = {'window_size': self.window_size,
                  'embedding_dims': self.embedding_dims}
        base_config = super(TimeEncoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# [SEQUENCE ENCODER]
#==============================================================================
# Custom sequence encoder
#==============================================================================
class SequenceEncoder(keras.layers.Layer):
    def __init__(self, window_size, kernel_size, embedding_dims, seed=42):
        super(SequenceEncoder, self).__init__()
        self.window_size = window_size        
        self.kernel_size = kernel_size
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.embedding = Embedding(input_dim=20, output_dim=embedding_dims)          
        self.conv1 = Conv2D(128, strides=(2,2), kernel_size=self.kernel_size, kernel_initializer='he_uniform', 
                            padding='same', activation='relu')         
        self.conv2 = Conv2D(256, strides=(2,2), kernel_size=self.kernel_size, kernel_initializer='he_uniform', 
                            padding='same', activation='relu')         
        self.reshape = Reshape((-1, 128))        
        self.lstm1 = LSTM(256, use_bias=True, return_sequences=True, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)
        self.lstm2 = LSTM(512, use_bias=True, return_sequences=False, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)
        self.dense1 = Dense(512, kernel_initializer='he_uniform', activation='LeakyReLU')
        self.BN = BatchNormalization()
        self.dropout = Dropout(rate=0.2, seed=self.seed)
        self.dense_output = Dense(256, kernel_initializer='he_uniform', activation='LeakyReLU')

    # implement sequence encoding through call method  
    #--------------------------------------------------------------------------   
    def call(self, inputs, training):
        layer = self.embedding(inputs)               
        layer = self.conv1(layer)       
        layer = self.conv2(layer)       
        layer = self.reshape(layer)
        layer = self.lstm1(layer)
        layer = self.lstm2(layer)
        layer = self.dense1(layer)
        layer = self.BN(layer, training)
        layer = self.dropout(layer, training)
        dense_output = self.dense_output(layer)

        return dense_output 

    # get configuration to print out structure
    #--------------------------------------------------------------------------
    def get_config(self):
        config = {'window_size': self.window_size,
                  'kernel_size': self.window_size,
                  'embedding_dims': self.embedding_dims,
                  'seed': self.seed}
        base_config = super(SequenceEncoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))   

# [SPECIAL ENCODER]
#==============================================================================
# Custom special number encoder
#==============================================================================
class SpecialEncoder(keras.layers.Layer):
    def __init__(self, window_size, embedding_dims, seed=42):
        super(SpecialEncoder, self).__init__()
        self.window_size = window_size               
        self.embedding_dims = embedding_dims
        self.seed = seed
        self.embedding = Embedding(input_dim=18, output_dim=self.embedding_dims)        
        self.reshape = Reshape((-1, self.embedding_dims)) 
        self.lstm1 = LSTM(64, use_bias=True, return_sequences=True, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)
        self.lstm2 = LSTM(128, use_bias=True, return_sequences=False, activation='tanh', 
                          kernel_initializer='glorot_uniform', dropout=0.2)
        self.BN = BatchNormalization()        
        self.dense_output = Dense(256, kernel_initializer='he_uniform', activation='LeakyReLU')

    # implement sequence encoding through call method  
    #--------------------------------------------------------------------------    
    def call(self, inputs, training):
        layer = self.embedding(inputs)        
        layer = self.reshape(layer)
        layer = self.lstm1(layer)
        layer = self.lstm2(layer)
        layer = self.BN(layer, training)       
        dense_output = self.dense_output(layer)

        return dense_output

    # get configuration to print out structure
    #--------------------------------------------------------------------------
    def get_config(self):
        config = {'window_size': self.window_size,                 
                  'embedding_dims': self.embedding_dims,
                  'seed': self.seed}
        base_config = super(SpecialEncoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))  
      
# [WFL DECODER]
#==============================================================================
# Custom cdecoder
#==============================================================================
class SequenceDecoder(keras.layers.Layer):
    def __init__(self, seed=42):
        super(SequenceDecoder, self).__init__()
        self.seed = seed
        self.add = Add()        
        self.BN1 = BatchNormalization()  
        self.BN2 = BatchNormalization()          
        self.dense1 = Dense(256, kernel_initializer='he_uniform', activation='relu')
        self.dense2 = Dense(128, kernel_initializer='he_uniform', activation='relu')
        self.dense3 = Dense(64, kernel_initializer='he_uniform', activation='relu')        
        self.seq_output = Dense(20, activation='sigmoid', dtype='float32')

    # implement decoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, encoder_seq, concat_output, training):
        concat = self.add([encoder_seq, concat_output])       
        sequence = self.dense1(concat)
        sequence = self.BN1(sequence, training)
        sequence = self.dense2(sequence)
        sequence = self.BN2(sequence, training)
        sequence = self.dense3(sequence)
        sequence = self.seq_output(sequence)       

        return sequence
    
    # get configuration to print out structure
    #--------------------------------------------------------------------------
    def get_config(self):
        config = {'seed': self.seed}
        base_config = super(SequenceDecoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# [WFL DECODER]
#==============================================================================
# Custom cdecoder
#==============================================================================
class SpecialDecoder(keras.layers.Layer):
    def __init__(self, seed=42):
        super(SpecialDecoder, self).__init__()
        self.seed = seed
        self.add = Add()          
        self.BN = BatchNormalization()  
        self.dense = Dense(128, kernel_initializer='he_uniform', activation='relu')
        self.special_output = Dense(20, activation='softmax', dtype='float32')        

    # implement decoder through call method  
    #--------------------------------------------------------------------------    
    def call(self, encoder_special, concat_output, training):
        concat = self.add([encoder_special, concat_output])       
        special = self.dense(concat)
        special = self.BN(special, training)     
        special = self.special_output(special)       

        return special
    
    # get configuration to print out structure
    #--------------------------------------------------------------------------
    def get_config(self):
        config = {'seed': self.seed}
        base_config = super(SpecialDecoder, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# [DEEP LEARNING MODEL]
#==============================================================================
# Multiclass model for training
#==============================================================================
class WFLMultiSeq():

    '''
    MultiSeqWFL - A custom model for multi-sequence time-series forecasting.

    Keyword Arguments:
        learning_rate (float): The learning rate for the optimizer.
        window_size (int): The size of the input window for the time series.
        time_embedding (int): The dimension of the time embedding.
        embedding_seq (int): The dimension of the sequence embedding.
        embedding_special (int): The dimension of the special number embedding.
        kernel_size (int): The size of the convolutional kernel.
        seed (int, optional): The seed for random initialization. Defaults to 42.
        XLA_state (bool, optional): Enable or disable XLA (Accelerated Linear Algebra) compilation. Defaults to False.

    Returns:
        tf.keras.Model: The compiled MultiSeqWFL model for multi-sequence time-series forecasting
                        (using method build() to compile the full model)

    '''
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
        self.time_encoder = TimeEncoder(window_size, time_embedding)
        self.sequence_encoder = SequenceEncoder(window_size, kernel_size, time_embedding, seed)
        self.special_encoder = SpecialEncoder(window_size, time_embedding, seed)           
        self.sequence_decoder = SequenceDecoder(seed) 
        self.special_decoder = SpecialDecoder(seed)  
   
    def build(self):     
        
        time_input = Input(shape=(self.window_size, 1))   
        sequence_input = Input(shape=(self.window_size, self.num_features))   
        special_input = Input(shape=(self.window_size, 1))        
        #----------------------------------------------------------------------
        time_head = self.time_encoder(time_input)        
        sequence_head = self.sequence_encoder(sequence_input)        
        special_head = self.special_encoder(special_input)
        concat = Concatenate()([time_head, sequence_head, special_head])        
        bnorm1 = BatchNormalization()(concat)
        densecat1 = Dense(512, kernel_initializer='he_uniform', activation='LeakyReLU')(bnorm1)
        bnorm2 = BatchNormalization()(densecat1) 
        densecat2 = Dense(384, kernel_initializer='he_uniform', activation='LeakyReLU')(bnorm2)
        bnorm3 = BatchNormalization()(densecat2) 
        densecat3 = Dense(256, kernel_initializer='he_uniform', activation='LeakyReLU')(bnorm3)        
        sequence_output = self.sequence_decoder(sequence_head, densecat3)
        special_output = self.special_decoder(special_head, densecat3)
        #----------------------------------------------------------------------        
        inputs = [time_input, sequence_input, special_input]
        outputs = [sequence_output, special_output]
        model = Model(inputs = inputs, outputs = outputs, name = 'MultiSeqWFL_model')    
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_ext = WFLossFunction(epsilon=1.0)
        loss_special = keras.losses.CategoricalCrossentropy(from_logits=False)
        metric_ext = keras.metrics.AUC(multi_label=True)
        metric_special = keras.metrics.CategoricalAccuracy()
        model.compile(optimizer = opt, loss = [loss_ext, loss_special],                       
                      metrics={'sequence_decoder': metric_ext, 
                               'special_decoder': metric_special},
                      jit_compile=self.XLA_state)       
        
        return model
   




# [MODEL TRAINING]
#==============================================================================
# Model training logic
#==============================================================================
class ModelTraining:

    '''
    ModelTraining - A class for configuring the device and settings for model training.

    Keyword Arguments:
        device (str):                         The device to be used for training. 
                                              Should be one of ['default', 'GPU', 'CPU'].
                                              Defaults to 'default'.
        seed (int, optional):                 The seed for random initialization. Defaults to 42.
        use_mixed_precision (bool, optional): Whether to use mixed precision for improved training performance.
                                              Defaults to False.
    
    '''           
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
    
    #--------------------------------------------------------------------------
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
    #--------------------------------------------------------------------------
    def load_pretrained_model(self, path, load_parameters=True):

        '''
        Load pretrained keras model (saved in folder) from the specified directory. 
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


# [MODEL VALIDATION]
#============================================================================== 
# model validation logic
#==============================================================================
class ModelValidation:

    '''
    ModelValidation - A class for model validation and performance evaluation.

    Keyword arguments:
        model (tf.keras.Model): The trained model for validation.

    '''
    def __init__(self, model):      
        self.model = model       
    
    # comparison of data distribution using statistical methods 
    #--------------------------------------------------------------------------     
    def WFL_confusion(self, Y_real, predictions, name, path, dpi=400):

        '''
        Generate and save a confusion matrix plot for model predictions.

        Keyword arguments:
            Y_real (array-like): True labels.
            predictions (array-like): Predicted labels.
            name (str): Name to be used in the saved file.
            path (str): Directory path to save the plot.
            dpi (int, optional): Dots per inch for the saved plot. Defaults to 400.

        '''         
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
    #--------------------------------------------------------------------------
    def plot_multi_ROC(Y_real, predictions, class_dict, path, dpi):

        '''
        Generate and save a multi-class ROC curve plot for model predictions.

        Keyword arguments:
            Y_real (array-like): True labels.
            predictions (array-like): Predicted labels.
            class_dict (dict): A dictionary mapping class indices to class labels.
            path (str): Directory path to save the plot.
            dpi (int): Dots per inch for the saved plot.

        '''    
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
            plt.plot(fpr[i], tpr[i], label=f'ROC curve of {list(class_dict.keys())[i]} (area = {roc_auc[i]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")       
        plot_loc = os.path.join(path, 'multi_ROC.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi=dpi)           
    

