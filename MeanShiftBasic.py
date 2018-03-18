##Mean shift is another clustering algorithm.
##Mean Shift assumes all data points as their own cluster centroids.
##It uses a radius-bandwidth to estimate data points in  its neighborhood and
##makes clusters. This happens for all points until they reach a point of convergence
##
##This is an example of Heirarchical clustering where we let the machine decide
##the number of clusters ad the data may give us insights we had not even thought about
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

centers= [[1,1,1], [5,5,5], [3,10,10]]   #center sample data around these
X, _= make_blobs(n_samples=100, centers=centers, cluster_std= 1)

ms= MeanShift()
ms.fit(X)
labels= ms.labels_
cluster_centers= ms.cluster_centers_
print(cluster_centers)    #cluster centers may vary from our centers as the centroid shifts
n_clusters_= len(np.unique(labels))
print("Number of estimated clusters: ", n_clusters_)

colors= 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig=plt.figure()
ax= fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='x')    #plot the points as 'x's

ax.scatter(cluster_centers[:,0], cluster_centers[:,1], cluster_centers[:,2],    #plot the 3 cluster centers
           marker='o', color='k', s=150, linewidth=5, zorder=10)

plt.show()










