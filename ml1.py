#CLASSIFICATION 

from sklearn.datasets import load_digits

digits = load_digits()

#print(digits.DESCR) #returns bunch obj, dictionaries with description
'''
print(digits.data[:2]) #each number represents pixel intensity btwn 0-16(light to dark), flatten array
print(digits.data.shape) #(rows-samples, columns-features)

print(digits.target[100:120]) #represents array for target sample

print(digits.target[:2]) #[0 1] with 0 representing the first data
print(digits.target.shape) #features are of a target already being represented

print(digits.images[:2]) #8x8 representation array of pixels
'''

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows = 4, ncols = 6, figsize = (6,4))#(6 image, across 4 rows)

#python zip function bundles the 3 iterables and produces one iterable
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item 
    axes.imshow(image, cmap = plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()

#plt.show()

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 11) 
#training set, testing set, sample and target in each set
#random shuffle has ensure that the sets have similar characteristics

print(x_train.shape) #75% goes towards training goes towards ##data 2D
print(y_train.shape) ##target for training 1D

print(x_test.shape) 
print(y_test.shape) #expected

from sklearn.neighbors import KNeighborsClassifier
#classify every sample in a class 0-9

knn = KNeighborsClassifier()

#load the training data into the model using the fit method
knn.fit(X = x_train, y = y_train)

#predict is from kneighbors
predicted = knn.predict(X = x_test) # no Y value bc youre looking for the Y value!!

expected = y_test

print(predicted[:20])
print(expected[:20])

#going through each element at the same time
wrong = [(p,e) for (p,e) in zip(predicted, expected) if p != e]
print(wrong)

#predict percentage accurracy
print(format(knn.score(x_test,y_test), ".2%"))

from sklearn.metrics import confusion_matrix

#gives an idea of what class(es) were predicted least accRTLY
cf = confusion_matrix(y_true = expected, y_pred = predicted)

print(cf)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

#create dataframe with 10  number 0-10
cf_df = pd.DataFrame(cf, index = range(10), columns = range(10))

fig = plt2.figure(figsize = (7,6))
axes = sns.heatmap(cf_df, annot = True, cmap = plt2.cm.nipy_spectral_r)
plt2.show()