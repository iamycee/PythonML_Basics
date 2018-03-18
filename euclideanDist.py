#Code to calculate euclidean distance and assign cluster to a new point
#Also plots the points on a graph

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import style
import warnings
style.use('fivethirtyeight')

dataset= {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}     #basically coordinates
#k-features, r- labels; two different classes/clusters
new_features=[5,7]

def kNearestNeighbors(data, predict, k=3):
    if len(data) >=k :
        warnings.warn('K is set to a value less than total number of clusters, idiot')
    distances= []
    for group in data:
        for features in data[group]:
            euclidean_distance= np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]    # first k of sorted's 1st element
    print(Counter(votes).most_common(1)[0])
    vote_result= Counter(votes).most_common(1)[0][0] #get most common votes

    return vote_result

result= kNearestNeighbors(dataset, new_features, k=3)    #assigning cluster to the new_features
print(result)

for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
plt.scatter(new_features[0], new_features[1], color= result, s=100)
plt.show()

