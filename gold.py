import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression


df = pd.read_csv("gold_price.csv", parse_dates=True, index_col='Date')
df.head()

df['Return'] = df['USD (PM)'].pct_change() * 100
df['Lagged_Return'] = df.Return.shift()
df = df.dropna()
df.to_csv('gold_clean.csv')

train = df['2001':'2018']
test = df['2019']
# Create train and test sets for dependent and independent variables
X_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
X_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]


'''
Using Linear Regression Model
Now as we have prepared the data to fit in a machine learning model for the task of gold price prediction, the next step is to choose a machine learning algorithm. For this task, I will use the Linear Regression algorithm:
'''
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

#plot the results of gold price prediction that we got from the linear regression algorithm

import matplotlib.pyplot as plt
out_of_sample_results = y_test.to_frame()
# Add a column of "out-of-sample" predictions to that dataframe:  
out_of_sample_results["Out-of-Sample Predictions"] = model.predict(X_test)
out_of_sample_results.plot(subplots=True, title='Gold prices, USD')
plt.show()
plt.savefig('gold_pred.png')


