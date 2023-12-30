# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 52
training_device = 'GPU'
epochs = 500
learning_rate = 0.001
batch_size = 1000

# embedding and convolutions
#------------------------------------------------------------------------------
embedding_time = 64
embedding_sequence = 128
embedding_special = 64
kernel_size = 6

# Define variables for preprocessing
#------------------------------------------------------------------------------
use_test_data = True
invert_test = False
data_size = 0.5
test_size = 0.1
window_size = 60
save_files = False

# k fold training
#------------------------------------------------------------------------------
k_fold = 12
k_epochs = 400

# Predictions variables
#------------------------------------------------------------------------------
predictions_size = 1000

# data collector variables
#------------------------------------------------------------------------------
headless = True
current_year = 2023
credentials = {'username' : 'p',
               'password' : 'p'}

