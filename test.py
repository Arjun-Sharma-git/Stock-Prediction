import subprocess
import sys

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
    from scipy.optimize import minimize
    import statsmodels.tsa.api as smt
    import statsmodels.api as sm
    from tqdm import tqdm_notebook
    from itertools import product

except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'seaborn'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'sklearn'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'scipy'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'statsmodels'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'tqdm'])
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'itertools'])

finally:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
    from scipy.optimize import minimize
    import statsmodels.tsa.api as smt
    import statsmodels.api as sm
    from tqdm import tqdm_notebook
    from itertools import product


# Getting the Data in string format from NodeJS
# NodeJS passed 2 args ['./test.py', data] where sys.argv[1] represents DATA
data = sys.argv[1]

# Creating a Pandas DataFrame for the supplied data 
my_data_new = pd.DataFrame([x.split(',') for x in data.split('\n')])

# Rearranging and Cleaning the Data 
my_data_new.columns = my_data_new.iloc[0]
my_data_new = my_data_new.drop(columns = ['dividend_amount', 'split_coefficient\r'])
my_data_new.drop(0, axis = 0, inplace = True)
my_data_new.drop(101, axis = 0, inplace = True)
my_data_new = my_data_new.iloc[::-1]        # reverse all rows
my_data_new.reset_index(drop = True, inplace = True)

# Converting the Data type of each data item like close, high, low etc from "Object" to "numeric"
for i in range(0, len(my_data_new.columns)):
    my_data_new.iloc[:,i] = pd.to_numeric(my_data_new.iloc[:,i], errors='ignore')


# Defining the Theme for Plotting work
plt.style.use('seaborn')

# Important Functions
# 1) this returns the mean absolute error b/w the predicted and the true value 
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 2) this function plots the moving average and the Required Plot
# accepts a pandas data frame named series => Required Plot
# accepts window size for rolling mean
# accepts name for the png figure to be save
def plot_moving_average(series, window, my_name):

    # creating a DataFrame for storing the rolling mean's
    rolling_mean = series.rolling(window=window).mean()

    # Plotting the Rolling Mean Data
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
   
           
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')  
    plt.grid(True)
    plt.xticks(rotation = 90)

    file_name = "public/fig_prediction_" + my_name + '.png'
    plt.savefig(file_name)
    
    # return the rolling mean DataFrame for the Current Dynamic(close, high, low etc) 
    # so as to complete the prediction process
    return rolling_mean

# 3) Prediction function for the Market Dynamic
# We pass last(x2,y2) and 2nd-last mean(x1, y1) and predict x2 + 1
# though we can use this for and point but it will increase error chance for values  
def my_prediction(x1, y1, x2, y2, X):
    slope = (y2 - y1) / (x2 - x1)
    y = y1 + slope * (X - x1)
    return (y, slope)

# 4) Market Mood Predicting Function
# uses Slope Value of the last 2 rolling mean values to predict the Up / Down Nature of the Prediction Plot
def mood(slope):
    if(slope >= 0):
        return "Up"
    else:
        return "Down"

# defining number of Rows and Columns into the DataFrame
r,c = my_data_new.shape

# result_prediction variable to store final result
result_prediction = ""

# ===============================   Generating Prediction =========================== 
window = 5
pleasePlot = sys.argv[2].split()
for dynamic in pleasePlot:
    if(dynamic == 'open'):
        # Plot the Prediction Plot
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.open)
        # plt.savefig('public/fig_opening_price.png')
        
        # Calculating the Prediction Value for Next Day
        rolling_mean = plot_moving_average(my_data_new.open, window, "opening")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'close'):
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.close)
        # plt.savefig('public/fig_closing_price.png')

        rolling_mean = plot_moving_average(my_data_new.close, window, "closing")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)
        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"
        
    elif(dynamic == 'high'):
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.high)
        # plt.savefig('public/fig_high_price.png')

        rolling_mean = plot_moving_average(my_data_new.high, window, "high")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # Adding the Results Obtained i.e. Predicted Value and Mood to "result_prediction"
        # using "%" as a separator
        result_prediction += str(prediction) + "%" + mood(slope) + "%"
        
    elif(dynamic == 'low'):
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.low)
        # plt.savefig('public/fig_low_price.png')

        rolling_mean = plot_moving_average(my_data_new.low, window, "low")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # adding the results to result_prediction
        result_prediction += str(prediction) + "%" + mood(slope) + "%"
        
    elif(dynamic == 'adjusted_close'):
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.adjusted_close)
        # plt.savefig('public/fig_adjusted_close_price.png')

        rolling_mean = plot_moving_average(my_data_new.adjusted_close, window, "adjusted_close")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # adding the results to result_prediction
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    elif(dynamic == 'volume'):
        # plt.figure(figsize=(20, 10))
        # plt.plot(my_data_new.timestamp, my_data_new.volume)
        # plt.savefig('public/fig_volume.png')

        rolling_mean = plot_moving_average(my_data_new.volume, window, "volume")
        y2 = rolling_mean.iloc[-1]
        x2 = r-1
        y1 = rolling_mean.iloc[-2]
        x1 = r-2
        prediction, slope = my_prediction(x1, y1, x2, y2, x2+1)

        # adding the results to result_prediction
        result_prediction += str(prediction) + "%" + mood(slope) + "%"

    else:
        print("No Market Dynamic Found to Plot")

# returning the results
print(result_prediction)

sys.stdout.flush