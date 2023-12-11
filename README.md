# WFL-forecasting

## Project description
This project is about predicting the results of Win For Life lottery based on previous extractions. Win for Life is a lottery game where the top prize is a lifetime annuity. This means that if you win the top prize, you will receive a certain amount of money regularly for the rest of your life! In the Win for Life game, you need to choose 10 numbers out of 20 and a "Numerone" (special number). The minimum play is 10 numbers + 1 Numerone (special number) at the price of 1 or 2 EUR. There are 8 categories of winnings and you can win up to 3,000 euros per month for 20 years. The game has an extraction every hour from 7 am to 11 pm, and if your combination matches the drawn numbers, you win. The amount you win depends on how many numbers you match. 

### MultiSeqWFL model
The MultiSeqWFL is a neural network model developed to predict the next Win for Life extractions (sequence of 10 numbers plus the special number) based on the lottery history from 2013 to the current year. The model architecture is based on 3 different inputs, namely the time information (date and daytime), the 10-point extraction sequences and the special numer timeseries. 

## How to use
Run the WLF_launcher.py file to launch the script and use the main menu to navigate the different options. The main menu will provide the user with the following options:

**1) Collect Win for Life extractions:** Connect to the website xamig.com to retrieve Win for Life history from 2013 to current year

**2) Analyze WFL data:** Perform timeseries analysis with statistical methods and data visualization     

**3) Pretrain WFL model:** Standard pretraining of the WFL forecast model using prefetched data

**4) Pretrain WFL model 8k-fold training):** Iterative k-fold pretraining of the WFL forecast model using prefetched data

**5) Predict next extraction:** Predict the next coming extraction using the pretrained model

**6) Exit and close**

### Configurations
The configurations.py file allows to change the script configuration. The following parameters are available:

- `generate_model_graph:` generate and save 2D model graph (as .png file)
- `use_mixed_precision:` whether or not to use mixed precision for faster training (mix float16/float32)
- `use_tensorboard:` activate or deactivate tensorboard logging
- `XLA_acceleration:` use of linear algebra acceleration for faster training 
- `training_device:` select the training device (CPU or GPU)
- `epochs:` number of training iterations
- `learning_rate:` learning rate of the model during training
- `batch_size:` size of batches to be fed to the model during training
- `embedding_size:` embedding dimensions (valid for both models)
- `use_test_data:` whether or not to use test data
- `invert_test:` test data placement (True to set last points as test data)
- `data_size:` fraction of total available data to use
- `test_size:` fraction of data to leave as test data
- `window_size:` length of the input timeseries window
- `output_size:` number of next points to predict (output sequence)
- `predictions_size:` number of timeseries points to take for the predictions inputs

### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `art==6.1`
- `keras==2.10.0`
- `matplotlib==3.7.2`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `scikit-learn==1.3.0`
- `seaborn==0.12.2`
- `selenium==4.11.2`
- `tensorflow==2.10.0`
- `tqdm==4.66.1`
- `webdriver-manager==4.0.1`

These dependencies are specified in the provided `requirements.txt` file to ensure full compatibility with the application. 
