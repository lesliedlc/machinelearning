#SIMPLE LINEAR REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split

#####AVG NYC JAN#####
nyc_jan = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")#create out of the file

print(nyc_jan.head(3))#prints out first three of df

#only need dates bc they are the feature
#print(nyc_jan.Date.values)#each column 1D
#print(nyc_jan.Date.values.reshape(-1,1)) #make the top 1D into a 2D (-1 rows as there are dates, 1 column)

#x data, y target
x_train, x_test, y_train, y_test = train_test_split(nyc_jan.Date.values.reshape(-1,1), nyc_jan.Temperature.values, random_state = 11)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X = x_train, y = y_train)#expects samplesa and targets for training

#print(lr.coef_) #slope m
#print(lr.intercept_) #y interecept b

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
    data = nyc_jan, 
    x='Date',
    y = 'Temperature',
    hue = 'Temperature', 
    palette = 'winter', 
    legend = False)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc_jan.Date.values), max(nyc_jan.Date.values)])
#print(x)

y = predict(x)
#print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)


#####AVG NYC YEARLY#####
nyc_year = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")

#clean the date values to only have the year
nyc_year['Date'] = nyc_year.Date.astype(str).str[:4].astype(int)  

print(nyc_year.head(3))

x2_train, x2_test, y2_train, y2_test = train_test_split(nyc_year.Date.values.reshape(-1,1), nyc_year.Value.values, random_state = 11)

lr2 = LinearRegression()

lr2.fit(X = x2_train, y = y2_train)

#print(lr2.coef_) #slope m
#print(lr2.intercept_) #y interecept b

predicted2 = lr2.predict(x2_test)
expected2 = y2_test

for (p,e) in zip(predicted2[::5], expected2[::5]): 
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = (lambda x: lr2.coef_ * x + lr2.intercept_) 

print(predict(2020))
print(predict(1890))
print(predict(2021))

axes = sns.scatterplot(
    data = nyc_year, 
    x='Date',
    y = 'Value',
    hue = 'Value', 
    palette = 'winter', 
    legend = False)

axes.set_ylim(10,70)

x2 = np.array([min(nyc_year.Date.values), max(nyc_year.Date.values)])

y2 = predict(x2)

line2 = plt.plot(x2,y2)

plt.show()