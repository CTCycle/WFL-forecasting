# WFL-forecasting

## Project description
This project is about predicting the results of Win For Life lottery based on previous extractions. Win for Life is a lottery game where the top prize is a lifetime annuity. This means that if you win the top prize, you will receive a certain amount of money regularly for the rest of your life! In the Win for Life game, you need to choose 10 numbers out of 20 and a "Numerone" (special number). The minimum play is 10 numbers + 1 Numerone at the price of 1 or 2 EUR. There are 8 categories of winnings and you can win up to 3,000 euros per month for 20 years. The game has an extraction every hour from 7 am to 11 pm, and if your combination matches the drawn numbers, you win. The amount you win depends on how many numbers you match. 

The script has been developed to predict the next extraction based on the lottery history, by leveraging a large volume of data using a complex convolutional and recurrent Neural Network. ....

## How to use
Run the WLF_launcher.py file to launch the script and use the main menu to navigate the different options. The main menu will provide the user with the following options:

**1) Collect Win for Life extractions:** ....

**2) Analyze WFL data:** ...     

**3) Pretrain forecasting model:**

**4) Predict next extraction:**

**5) Exit and close**


### Requirements
This application has been developed and tested using the following dependencies (Python 3.10.12):

- `keras==2.10.0`
- `matplotlib==3.7.2`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `scikit-learn==1.3.0`
- `seaborn==0.12.2`
- `tensorflow==2.10.0`
- `xlrd==2.0.1`
- `XlsxWriter==3.1.3`
- `pydot==1.4.2`
- `graphviz==0.20.1`

These dependencies are specified in the provided `requirements.txt` file to ensure full compatibility with the application. 
