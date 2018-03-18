#Last time  we did kNearestNeighbors with the inbuilt sklearn neighbors classifier.
#In this one, we do it by writing our own code

import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def kNearestNeighbors(data, predict, k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total number of clusters, idiot')
    distances= []
    for group in data:
        for features in data[group]:
            euclidean_distance= np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]    #1st element is group, go upto k
    # print(Counter(votes).most_common(1)[0])
    vote_result= Counter(votes).most_common(1)[0][0] #get most common votes

    return vote_result

df= pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace= True)    #replacing NaNs
df.drop(['id'], 1, inplace=True)    #drop the id column, redundant data
full_data= df.astype(float).values.tolist()    #conversion to float; Use .values to get a numpy.array and then .tolist() to get a list.

#this is our train-test-split
random.shuffle(full_data)
test_size= 0.2
train_set= {2: [], 4:[]}    #2 indicates benign;  4 indicates benign
test_set= {2:[],  4:[]}
train_data= full_data[:-int(test_size*len(full_data))]    #everything upto the last 20% of data i.e 80% of the full_data is train_data
test_data= full_data[-int(test_size*len(full_data)):]   #last 20% of the data

#Creating the training and test set. Filling in values to the dicts above^^
for i in train_data:
    train_set[i[-1]].append(i[:-1])    #append the first n-1 elements to the last element of 2:[] and 4:[]; i.e appending to '[]'
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct= 0
total=0

#testing accuracy on train set
for group in test_set:
    for data in test_set[group]:
        vote= kNearestNeighbors(train_set, data, k=5)
        if group == vote:
            correct +=1    #as we know the groups for training set, we increment correct
        total +=1

print('Accuracy: ', correct/total)
















