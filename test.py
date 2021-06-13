import subprocess
import sys

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
    import pandas as pd




my_data = sys.argv[1]
mydata_temp = []
mydata_temp.append(my_data.split(','))
print(mydata_temp)
# my_data = my_data.drop(columns = ['dividend_amount', 'split_coefficient'])
#
# my_data_new = my_data.iloc[::-1]
#
# my_data_new = my_data_new.reset_index(drop = True)
# # output1 = my_data_new.to_json();
#
# def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
#
#     rolling_mean = series.rolling(window=window).mean()
#
#
#     #Plot confidence intervals for smoothed values
#     if plot_intervals:
#         mae = mean_absolute_error(series[window:], rolling_mean[window:])
#         deviation = np.std(series[window:] - rolling_mean[window:])
#         lower_bound = rolling_mean - (mae + scale * deviation)
#         upper_bound = rolling_mean + (mae + scale * deviation)
#     #plt.show()
#
#     return rolling_mean
#
# # Smooth by the previous 5 days (by week)
# rolling_mean = plot_moving_average(my_data_new.close, 5, True)
# output2 = rolling_mean.to_json()
# # print(output1)
sys.stdout.flush
