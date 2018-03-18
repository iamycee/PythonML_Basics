import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X= np.array([[1,2], [1.5, 1.8], [5,8], [8,8], [1,0.6], [9,11]])

##plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)    #all 0th elements and all 1st elements of X
##plt.show()

clf= KMeans(n_clusters=2)    #make 2 clusters
#Experiment with the number of clusters and see how the output varies
clf.fit(X)

centroids= clf.cluster_centers_
labels= clf.labels_

colors= ["g.", "r.", "c.", "b.", "k.", "o"]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=20)    #labels[i] returns either a 0 or a 1 as we have only 2 clusters 
    #Plots X and Y values,; 6 points- 3 red, 3 green.
    
plt.scatter(centroids[:,0],  centroids[:,1], marker='x', s=150, linewidth=5)
#plots the final 2 cluster centroids with 'x' sign
plt.show()



























