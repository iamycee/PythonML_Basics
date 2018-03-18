import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

X= np.array([[1,2], [1.5,1.8], [5,8], [8,8], [1,0.6], [9,11], [8,2], [10,2], [9,3]])

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors= 10*["g","r","c","b","k"]

#___MEAN SHIFT___#
# 1. Assign every single data point as a cluster center
# 2. Take data points within each cluster center's radius(bandwidth),
#    take the mean of all these datapoints and get a  new cluster center
# 3. Repeat step 2 until you get convergence.

class MeanShift:
    def __init__(self, radius=12):    #We have to manually set radius in this case
        self.radius= radius

    def fit(self, data):
        centroids= {}

        for i in range(len(data)):
            centroids[i]= data[i]    #set each point as a centroid

        while True:
            new_centroids= []   #whenever we find new centroids, we add them here
            for i in centroids:
                in_bandwidth=[]     #all points in BW of current centroid to be added here
                centroid= centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:    #see if it is within the set radius
                        in_bandwidth.append(featureset)

                new_centroid= np.average(in_bandwidth, axis=0)    #axis=0 means average over ALL the values
                new_centroids.append(tuple(new_centroid))

            #set takes only unique values; "sort the list version of these unique values"
            uniques= sorted(list(set(new_centroids)))

            prev_centroids= dict(centroids)

            centroids= {}
            for i in range(len(uniques)):
                centroids[i]= np.array(uniques[i])    #store these unique values in the centroids list
            optimized= True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):    #if not the same i.e if centroid has moved 
                    optimized= False    #if centroid has moved, it means that alg is not optimized yet
                if not optimized:
                    break
            if not optimized: 
                break
        #end while

        self.centroids= centroids

        def predict(self, data):
            pass

clf= MeanShift()
clf.fit(X)
centroids=clf.centroids
print(centroids)

plt.scatter(X[:,0], X[:,1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0],  centroids[c][1], color='k', marker='*', s=150)

plt.show()
















                    
                
