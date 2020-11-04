#SIMPLE LINEAR REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split

#create out of the file
nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

#prints out first three of df
print(nyc.head(3))

#only need dates bc they are the feature
print(nyc.Date.values)#each column 1D
print(nyc.Date.values.reshape(-1,1)) #make the top 1D into a 2D (-1 rows as there are dates, 1 column)

#x data, y target
x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values, random_state = 1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

#expects samplesa and targets for training
lr.fit(X = x_train, y = y_train)

print(lr.coef_) #slope m
print(lr.intercept_) #y interecept b

predicted = lr.predict(x_test)
expected = y_test

for (p,e) in zip(predicted[::5], expected[::5]): #2colon checks every 5th element in array
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = (lambda x: lr.coef_ * x + lr.intercept_) #kinda mx+b formula

print(predict(2020))
print(predict(1890))
print(predict(2021))

import seaborn as sns

axes = sns.scatterplot(
    data = nyc, 
    x='Date',
    y = 'Temperature',
    hue = 'Temperature', 
    palette = 'winter', 
    legend = False)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])
print(x)

y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)
plt.show()