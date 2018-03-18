import numpy as np
from sklearn  import preprocessing as pp, neighbors
from sklearn import model_selection as cv
import pandas as pd

df= pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace= True)  #dataset has 16 ?s, replace them;  can aslo do df.dropNa

#Getting rid of useless data like ID:
df.drop(['id'], 1, inplace= True)

X= np.array(df.drop(['class'], 1))  #everything except labels is the features
y= np.array(df['class'])    #y is the labelss

X_train, X_test, y_train, y_test= cv.train_test_split(X, y, test_size= 0.2)  #20% testing data
clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,  y_train)

accuracy= clf.score(X_test, y_test)
print(accuracy)

example_measures= np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
example_measures=example_measures.reshape(1,-1)
prediction= clf.predict(example_measures)
print(prediction)
