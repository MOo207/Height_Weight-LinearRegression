import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

## Linear Regression Implementation predicting weight given height
# y^ = theta_0 + theta_1 * x
filename = r'.\HeightWeightDataSet.csv'
heightWeightDataSet = pd.read_csv(filename)

# printing heightWeightDataSet data
print('--------All Dataset------------')
print(heightWeightDataSet)
print('---------Head-----------')
print(heightWeightDataSet.head())
print('----------Tail----------')
print(heightWeightDataSet.tail())
print('----------Shape----------')
print(heightWeightDataSet.shape)
# %%

# Returns : DataFrame Mask of bool values for each element in DataFrame
# that indicates whether an element is not an NA value.
print(heightWeightDataSet.isna())

# DataFrame.any()
# Return whether any element is True, potentially over an axis.
# Returns False unless there at least one element within a series or
# along a Dataframe axis that is True or equivalent (e.g. non-zero or non-empty).
print(heightWeightDataSet.isna().any())

## Create Tuple With One Item
## To create a tuple with only one item, you have to add a comma after the item,
## otherwise Python will not recognize it as a tuple.
x2 = heightWeightDataSet['Height'].to_numpy()  # ==> x2 This is an array of shape (n,)
print(x2.shape)

y = heightWeightDataSet['Weight'].to_numpy()
print(y.shape)

## Reshape 1D to 2D Array
## It is common to need to reshape a one-dimensional array
## into a two-dimensional array with one column and multiple rows.
sh=x2.shape
x = x2.reshape(sh[0], 1)  # ==>  x this is an n*1 matrix of shape (n,1)
print(x.shape);

## Train_test_split
## the first parameter X must be a matrix of at least n*1 dimension
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)  #<== this function performs the training 



print("The intercept is: theta_0=", linreg.intercept_)
print("The coefficient of the features are: theta_1 = ", linreg.coef_)
#The intercept is: theta_0= -226.2530611479022
#The coefficient of the features are: theta_1 =  [5.98632569]

# Obtain the error
Y_pred = linreg.predict(X_test)
MSE = mean_squared_error(Y_test, Y_pred)
print('Mean Square Error', MSE)

# Plotting the model and data
plt.scatter(X_test, Y_test, color='b', marker='x')
plt.plot(X_test, Y_pred, color='k')
plt.show()