# Define general variables
#------------------------------------------------------------------------------
generate_model_graph = True
use_mixed_precision = True
use_tensorboard = False
XLA_acceleration = False

# Define variables for the training
#------------------------------------------------------------------------------
seed = 42
training_device = 'GPU'
neuron_baseline = 56
epochs = 1
learning_rate = 0.005
batch_size = 600

# embedding and convolutions
#------------------------------------------------------------------------------
embedding_sequence = 512
embedding_special = 256
kernel_size = 6

# Define variables for preprocessing
#------------------------------------------------------------------------------
use_test_data = True
invert_test = False
data_size = 0.2
test_size = 0.2
window_size = 60
save_files = False

# Predictions variables
#------------------------------------------------------------------------------
predictions_size = 1000

# data collector variables
#------------------------------------------------------------------------------
headless = False
current_year = 2023
credentials = {'username' : 'st.gilles98@gmail.com',
               'password' : 'Fg566wrùF'}

