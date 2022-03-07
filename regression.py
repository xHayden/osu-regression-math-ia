import ast
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

x = []
y = []
lables = []
data = []

## Collect data from file
with open('osu_top_players.txt', 'r') as f:
        text = f.read()
        text = ast.literal_eval(text)
        for player in text:
            data.append({"playcount": player['playcount'], "pp": player['pp'], "username": player['username']})
            x.append(player['playcount'])
            y.append(player['pp'])
            lables.append(player['username'])

## Create reject outliers function
def reject_outliers(data, threshold=2.5, variable=None):
    median = np.median(data); # original median of the data
    print("Median for " + variable + ": " + str(median))
    diff = np.abs(data - median); # difference from the median
    mdev = np.median(diff); # median absolute deviation
    print("Median absolute deviation for " + variable + ": " + str(mdev))
    print("2.5 * MAD for " + variable + ": " + str(mdev * threshold))
    print("Upper bound for " + variable + ": " + str(median + (mdev * threshold)))
    print("Lower bound for " + variable + ": " + str(median - (mdev * threshold)))
    return data[diff < mdev * threshold]; # return the data that is within the threshold

## Removes outliers
removed_outliers_x = reject_outliers(np.asarray(x), 10, "x")
removed_outliers_y = reject_outliers(np.asarray(y), 10, "y")
new_x = [];
new_y = [];
for (i, player) in enumerate(data):
    if player['playcount'] not in removed_outliers_x or player['pp'] not in removed_outliers_y:
        data.pop(i)
        continue
    new_x.append(player['playcount']);
    new_y.append(player['pp']);
x = new_x;
y = new_y;

## Create least squares regression function
mean_x = np.mean(x)
mean_y = np.mean(y)
diff_x = []
diff_y = []
diff_x_squared = []
diff_y_squared = []
diff_x_times_diff_y = []
sum_x_2 = 0;
sum_diff_x_times_diff_y = 0;
for i in range(len(x)):
    diff_x.append(x[i] - mean_x)
    diff_y.append(y[i] - mean_y)
    diff_x_squared.append(diff_x[i]**2)
    diff_y_squared.append(diff_y[i]**2)
    diff_x_times_diff_y.append(diff_x[i] * diff_y[i])
    sum_diff_x_times_diff_y += diff_x_times_diff_y[i]
    sum_x_2 += diff_x_squared[i]
b = sum_diff_x_times_diff_y/sum_x_2
a = -1 * b * mean_x + mean_y

def predict_with_least_squares_lin_reg(x):
    return a + b * x
predicted_y = []
for i in range(len(x)):
    predicted_y.append(predict_with_least_squares_lin_reg(x[i]))
print("y = " + str(a) + " + " + str(b) + "x")

## Get R^2 value
sum_of_squared_error = 0;
sum_of_squared_regression = 0;

for i in range(len(x)):
    sum_of_squared_error += (y[i] - predicted_y[i])**2
    sum_of_squared_regression += (predicted_y[i] - mean_y)**2

sum_of_squared_total = sum_of_squared_error + sum_of_squared_regression;
r_squared = 1 - (sum_of_squared_error/sum_of_squared_total)
print("R^2 = " + str(r_squared))

## Plot linear regression
figure(figsize=(8, 6), dpi=200)
plt.ylabel('Performance Points (pp)')
plt.xlabel('Playcount')
plt.plot(x, y, 'ro', markersize=0.5)
plt.plot(x, predicted_y, 'b')
plt.show()
plt.xlim(0, 1200000)
plt.ylim(0, 30000)