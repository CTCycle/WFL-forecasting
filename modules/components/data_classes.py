import os
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ...
#==============================================================================
#==============================================================================
#==============================================================================
class UserOperations:
    
    """    
    A class for user operations such as interactions with the console, directories 
    and files cleaning and other maintenance operations.
      
    Methods:
        
    menu_selection(menu):         console menu management
    clear_all_files(folder_path): cleaning files and directories 
   
    """
    
    #==========================================================================
    def menu_selection(self, menu):
        
        """        
        menu_selection(menu)
        
        Presents a menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        """        
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print('{0} - {1}'.format(key, value))       
        print()
        while True:
            try:
                op_sel = int(input('Select the desired operation: '))
            except:
                continue           
            while op_sel not in indexes:
                try:
                    op_sel = int(input('Input is not valid, please select a valid option: '))
                except:
                    continue
            break
        
        return op_sel    
    
               
    #==========================================================================
    def datetime_fetching(self):
        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-7]
        today_datetime = truncated_datetime
        for rep in ('-', ':', ' '):
            today_datetime = today_datetime.replace(rep, '_')
            
        return today_datetime
      

# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class PreProcessing:
           

    #==========================================================================
    def dataset_composer(self, paths):
        
        '''
        '''
        dataframe_list = []
        with open(paths[0], 'r') as data:
                columns_string = [line.strip() for line in data][2]  
                self.columns = columns_string.split()              
        for pt in paths:
            with open(pt, 'r') as data:
                text = [line.strip() for line in data]
                extractions = [x.split() for x in text[3:]]
                dataframe = pd.DataFrame(extractions, columns=self.columns)                       
                dataframe_list.append(dataframe)            

        all_extractions = pd.concat(dataframe_list)
        all_extractions['Data'] = pd.to_datetime(all_extractions['Data'], dayfirst=True)   
        sorted_extractions = all_extractions.sort_values(['Data', 'Ora'])                  

        return sorted_extractions    
    
    
    # Splits time series data into training and testing sets using TimeSeriesSplit
    #==========================================================================
    def split_timeseries(self, dataframe, test_size, inverted=False):
        
        """
        timeseries_split(dataframe, test_size)
        
        Splits the input dataframe into training and testing sets based on the test size.
    
        Keyword arguments:  
        
        dataframe (pd.dataframe): the dataframe to be split
        test_size (float):        the proportion of data to be used as the test set
    
        Returns:
            
        df_train (pd.dataframe): the training set
        df_test (pd.dataframe):  the testing set
        
        """
        train_size = int(len(dataframe) * (1 - test_size))
        test_size = len(dataframe) - train_size 
        if inverted == True:
            df_train = dataframe.iloc[:test_size]
            df_test = dataframe.iloc[test_size:]
        else:
            df_train = dataframe.iloc[:train_size]
            df_test = dataframe.iloc[train_size:]

        return df_train, df_test    
    
    # generate n real samples with class labels; We randomly select n samples 
    # from the real data array
    #========================================================================== 
    def timeseries_labeling(self, df, window_size):
        
        """
        timeseries_labeling(dataframe, window_size)
    
        Labels time series data by splitting into input and output sequences using sliding window method.
    
        Keyword arguments:
            
        dataframe (pd.DataFrame): the time series data to be labeled
        window_size (int):        the number of time steps to use as input sequence
    
        Returns:
            
        X_array (np.ndarray):     the input sequence data
        Y_array (np.ndarray):     the output sequence data
        
        """        
        label = np.array(df)               
        X = [label[i : i + window_size] for i in range(len(label) - window_size)]
        Y = [label[i + window_size] for i in range(len(label) - window_size)]
        
        return np.array(X), np.array(Y)        
        
    #==========================================================================
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        model_name = f'{model_name}_{today_datetime}'
        model_savepath = os.path.join(path, model_name)
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)               
            
        return model_savepath
    
# define class for correlations calculations
#==============================================================================
#==============================================================================
#==============================================================================
class MultiCorrelator:
    
    """    
    Calculates the correlation matrix of a given dataframe using specific methods.
    The internal functions retrieves correlations based on Pearson, Spearman and Kendall
    methods. This class is also used to plot the correlation heatmap and filter correlations
    from the original matrix based on given thresholds. Returns the correlation matrix
    
    Keyword arguments:         
        dataframe (pd.dataframe): target dataframe
    
    Returns:
        df_corr (pd.dataframe): correlation matrix in dataframe form
                
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    # Spearman correlation calculation
    #--------------------------------------------------------------------------
    def Spearman_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'spearman').round(decimals)
        return self.df_corr
    
    # Kendall correlation calculation
    #--------------------------------------------------------------------------    
    def Kendall_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'kendall').round(decimals)
        return self.df_corr
    
    # Pearson correlation calculation
    #--------------------------------------------------------------------------    
    def Pearson_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'pearson').round(decimals)
        return self.df_corr
    
    # plotting correlation heatmap using seaborn package
    #--------------------------------------------------------------------------
    def corr_heatmap(self, matrix, path, dpi, name):
        
        """        
        Plot the correlation heatmap using the seaborn package. The plot is saved 
        in .jpeg format in the folder that is specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:           
            matrix (pd.dataframe): target correlation matrix
            path (str):            picture save path for the .jpeg file
            dpi (int):             value to set picture quality when saved (int)
            name (str):            name to be added in title and filename
        
        Returns:            
            None
            
        """
        cmap = 'YlGnBu'
        sns.heatmap(matrix, square = True, annot = False, 
                    mask = False, cmap = cmap, yticklabels = False, 
                    xticklabels = False)
        plt.title('{}_correlation_heatmap'.format(name))
        plt.tight_layout()
        plot_loc = os.path.join(path, '{}_correlation_heatmap.jpeg'.format(name))
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show(block = False)
        plt.close()
        
    # plotting correlation heatmap of two dataframes
    #--------------------------------------------------------------------------
    def double_corr_heatmap(self, matrix_real, matrix_fake, path, dpi):        
        
        """        
        Plot the correlation heatmap of two dataframes using the seaborn package. 
        The plot is saved in .jpeg format in the folder specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:
            matrix_real (pd.dataframe): real data correlation matrix
            matrix_fake (pd.dataframe): fake data correlation matrix
            path (str):                 picture save path for the .jpeg file
            dpi (int):                  value to set picture quality when saved (int)
        
        Returns:
            None

        """ 
        plt.subplot(2, 1, 1)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(matrix_real, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Real data')
        plt.subplot(2, 1, 2)
        sns.heatmap(matrix_fake, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Synthetic data')
        plt.tight_layout()
        plot_loc = os.path.join(path, 'Correlation_heatmap.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show()
        plt.close()
     
    # filtering of correlation pairs based on threshold value. Strong, weak and null
    # pairs are isolated and embedded into output lists
    #--------------------------------------------------------------------------    
    def corr_filter(self, matrix, threshold): 
        
        """       
        Generates filtered lists of correlation pairs, based on the given threshold.
        Weak correlations are those below the threshold, strong correlations are those
        above the value and zero correlations identifies all those correlation
        with coefficient equal to zero. Returns the strong, weak and zero pairs lists
        respectively.
        
        Keyword arguments:    
            matrix (pd.dataframe): target correlation matrix
            threshold (float):     threshold value to filter correlations coefficients
        
        Returns:            
            strong_pairs (list): filtered strong pairs
            weak_pairs (list):   filtered weak pairs
            zero_pairs (list):   filtered zero pairs
                       
        """        
        self.corr_pairs = matrix.unstack()
        self.sorted_pairs = self.corr_pairs.sort_values(kind="quicksort")
        self.strong_pairs = self.sorted_pairs[(self.sorted_pairs) >= threshold]
        self.strong_pairs = self.strong_pairs.reset_index(level = [0,1])
        mask = (self.strong_pairs.level_0 != self.strong_pairs.level_1) 
        self.strong_pairs = self.strong_pairs[mask]
        
        self.weak_pairs = self.sorted_pairs[(self.sorted_pairs) >= -threshold]
        self.weak_pairs = self.weak_pairs.reset_index(level = [0,1])
        mask = (self.weak_pairs.level_0 != self.weak_pairs.level_1) 
        self.weak_pairs = self.weak_pairs[mask]
        
        self.zero_pairs = self.sorted_pairs[(self.sorted_pairs) == 0]
        self.zero_pairs = self.zero_pairs.reset_index(level = [0,1])
        mask = (self.zero_pairs.level_0 != self.zero_pairs.level_1) 
        self.zero_pairs_P = self.zero_pairs[mask]
        
        return self.strong_pairs, self.weak_pairs, self.zero_pairs

    
# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class TimeSeriesAnalysis:    
    
    
    # comparison of histograms (distributions) by superimposing plots
    #========================================================================== 
    def timeseries_histogram(self, dataframe, bins, path, dpi=400):
        
        """       
        Plots a histogram of the given time series data and saves it as a JPEG image.
    
        Keyword arguments:            
            timeseries (pd.DataFrame): Time series data.
            bins (int):                Number of histogram bins.
            name (str):                Name of the time series.
            path (str):                Path to save the histogram figure.
            dpi (int):                 DPI for the JPEG image.
    
        Returns:            
            None
        
        """ 
        num_cols = len(dataframe.columns)
        num_rows = num_cols // 2 if num_cols % 2 == 0 else num_cols // 2 + 1
        fig, axs = plt.subplots(num_rows, 2, figsize=(10, num_rows*5))
        axs = axs.flatten()  # to handle the case where num_cols < 2

        for i, col in enumerate(dataframe.columns):
            array = dataframe[col].values
            axs[i].hist(array, bins = bins, density = True, label = col,
                        rwidth = 1, edgecolor = 'black')        
            axs[i].legend(loc='upper right')
            axs[i].set_title('Histogram of {}'.format(col))
            axs[i].set_xlabel('Time', fontsize = 8)
            axs[i].set_ylabel('Norm frequency', fontsize = 8) 
            axs[i].tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plot_loc = os.path.join(path, 'histogram_timeseries.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)         

        plt.close(fig)     


        
        
    # comparison of histograms (distributions) by superimposing plots
    
    
    
    # comparison of data distribution using statistical methods 
    #========================================================================== 
    # def test_stationarity(self, timeseries, name, path, dpi):
            
    #     ADF = adfuller(timeseries)                    
    #     fig, ax = plt.subplots(3, sharex = True, sharey = False)                  
    #     ax[0].set_title('Original')
    #     ax[1].set_title('Rolling Mean')
    #     ax[2].set_title('Rolling Standard Deviation')
    #     plt.xlabel('Time', fontsize = 8)
    #     plt.xticks(fontsize = 8)
    #     plt.yticks(fontsize = 8)
    #     plt.tight_layout()                       
    #     plot_loc = os.path.join(path, 'timeseries_analysis_{}.jpeg'.format(name))
    #     plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
    #     #plt.show(block = False)
        
    #     return ADF, fig
          
    # comparison of data distribution using statistical methods 
    #==========================================================================     
    def predictions_vs_test(self, Y_test, predictions, path, dpi):       
            
        plt.figure()
        plt.plot(Y_test, color='blue', label = 'test values')
        plt.plot(predictions, color='red', label = 'predictions')
        plt.legend(loc='best')
        plt.title('Test values vs Predictions')
        plt.xlabel('Time', fontsize=8)
        plt.ylabel('Values', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plot_loc = os.path.join(path, 'test_vs_predictions.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi = dpi)
        plt.show(block=False)
     
    
    
   
     
