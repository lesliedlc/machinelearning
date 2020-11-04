#MULTIPLE LINEAR REGRESSION

from sklearn.datasets import fetch_california_housing

cali = fetch_california_housing() #bunch object

#print(cali.DESCR)

print(cali.data.shape) #(rows,columns)

print(cali.target.shape) #identifier for each row

print(cali.feature_names) #names of columns

import pandas as pd
pd.set_option("precision", 4)
pd.set_option("max_columns", 9) #display 9 columns
pd.set_option("display.width", None) #autodetct width

cali_df = pd.DataFrame(cali.data, columns = cali.feature_names) #data

cali_df["MedianHouseValue"] = pd.Series(cali.target) #new column as series to be added to df

print(cali_df.head())

sample_df = cali_df.sample(frac = .1, random_state = 17)

import matplotlib.pyplot as plt
import seaborn as sns #mapping

#settings
sns.set(font_scale = 2)
sns.set_style("whitegrid")

for feature in cali.feature_names:
    plt.figure(figsize=(8, 4.5)) #8 by 4.5" figure
    sns.scatterplot( 
        data = sample_df,
        x = feature,
        y = "MedianHouseValue",
        hue = "MedianHouseValue",
        palette = "cool",
        legend = False,
    )

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

train_data, test_data, train_target, test_target = train_test_split(cali.data, cali.target, random_state = 11)

lr.fit(X = train_data, y = train_target)

predicted = lr.predict(test_data)
expected = test_target

print(f"predicted:{predicted[::5]} expected: {expected[::5]}")

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)
df["Predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize = (9, 9))

axes = sns.scatterplot( 
        data = df,
        x = "Expected",
        y = "Predicted",
        hue = "Predicted",
        palette = "cool",
        legend = False,
    )

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
print(start)
print(end)

axes.set_xlim(start,end)
axes.set_ylim(start,end)

line = plt2.plot([start,end],[start,end],"k--" ) #black dotted lines = k--
plt2.show()