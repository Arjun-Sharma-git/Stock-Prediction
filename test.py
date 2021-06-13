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

# import warnings
# warnings.filterwarnings('ignore')


data = sys.argv[1]
my_data_new = pd.DataFrame([x.split(',') for x in data.split('\n')])
# print(my_data_new.columns)

# my_data_list_tmp = my_data_str.split('\r')      # breaking into list
# my_data_list_tmp.pop()

# my_data_list_tmp2 = []          # this will break each list to form rows and columns
# for st in my_data_list_tmp:
#     my_st = st.split(',')

#     my_data_list_tmp2.append(my_st);

# R = len(my_data_list_tmp2)
# C = len(my_data_list_tmp2[0])
# for i in range(1,R):
#     prev_text = my_data_list_tmp2[i][0]

#     # replace the \n from the dates
#     new_text = prev_text[1:]
#     my_data_list_tmp2[i][0] = my_data_list_tmp2[i][0].replace(prev_text, new_text) 

#     # changing the data types of the rest columns other than dates to float 
#     for j in range(1,C):
#         my_data_list_tmp2[i][j] = float(my_data_list_tmp2[i][j])

# my_data_new = pd.DataFrame(my_data_list_tmp2)
# # print(my_data_new.shape)

my_data_new.columns = my_data_new.iloc[0]
my_data_new = my_data_new.drop(columns = ['dividend_amount', 'split_coefficient\r'])
my_data_new.drop(0, axis = 0, inplace = True)
my_data_new.drop(101, axis = 0, inplace = True)
my_data_new = my_data_new.iloc[::-1]        # reverse all rows
my_data_new.reset_index(drop = True, inplace = True)

for i in range(0, len(my_data_new.columns)):
    my_data_new.iloc[:,i] = pd.to_numeric(my_data_new.iloc[:,i], errors='ignore')


# Plotting work
plt.style.use('seaborn')

plt.figure(figsize=(20, 10))
plt.plot(my_data_new.timestamp, my_data_new.close)
plt.savefig('fig_closing_price.png')

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_moving_average(series, window, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')
   
    #Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
           
    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='best')  
    plt.grid(True)
    plt.xticks(rotation = 90)
    plt.savefig('fig_prediction.png')
    #plt.show()
    
    return rolling_mean
   
# Smooth by the previous 5 days (by week)
rolling_mean = plot_moving_average(my_data_new.close, 5, True)

# prediction
r,c = my_data_new.shape
y2 = rolling_mean.iloc[-1]
x2 = r-1
y1 = rolling_mean.iloc[-2]
x1 = r-2

def prediction_closing_price(x1, y1, x2, y2, X = x2 + 1):
    slope = (y2 - y1) / (x2 - x1)
    y = y1 + slope * (X - x1)
    return (y, slope)

prediction, slope = prediction_closing_price(x1, y1, x2, y2)

# returning the results

market_mood = ""
if(slope >= 0):
    market_mood = "Up"
else:
    market_mood = "Down"

result_string = str(prediction) + "%" + market_mood

print(result_string)

sys.stdout.flush