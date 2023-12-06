import os

# Define paths
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
fc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictions')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
pp_path = os.path.join(data_path, 'preprocessed data')
als_path = os.path.join(data_path, 'analysis')

# Create folders
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(pp_path):
    os.mkdir(pp_path)
if not os.path.exists(fc_path):
    os.mkdir(fc_path)
if not os.path.exists(als_path):
    os.mkdir(als_path)

