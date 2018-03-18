import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
#from sklearn.cluster import KMeans <- deleted as we are making our own


X= np.array([[1,2], [1.5, 1.8], [5,8], [8,8], [1,0.6], [9,11]])

plt.scatter(X[:,0], X[:,1], s=150, linewidths=5)    #all 0th elements and all 1st elements of X
plt.show()
#plots original data, without clustering obviously

colors= 10*["g", "r", "c", "b", "k", "o"]

##KMeans build starts
##What K-means clustering does is that it takes 2 random points as initial centroids
##1. Find euclidean distance of all points and assign each point to either of the
##   two centroids
##2. With the 2 clusters, find their mean, take this mean location as new centroid
##   and perform the cluster assignment step 1
##3. Repeat steps 1 and 2 until the location of the centroid has stopped moving
class KMeans:
    def __init__(self, k=2, tol= 0.001, max_iter=300):
        self.k= k
        self.tol=tol
        self.max_iter= max_iter
    def fit(self, data):

        self.centroids= {}    #empty on every iteration as centroid chages location
        #this dict contains another dict, thus is reference

        for i in range(self.k):
            self.centroids[i]=data[i]    #set first 2 data points as first 2 centroids

        for i in range(self.max_iter):
            self.classifications= {}    #used to store which cluster is our featureset classified into

            for i in range(self.k):
                self.classifications[i]= []

            for featureset in data:
                #list of distances of each point in the featureset to the current centroids
                distances= [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification= distances.index(min(distances))    #index 0/1 of the minimum distance
                self.classifications[classification].append(featureset)    #if 0 then append to 0, if 1 then append to 1
                
            prev_centroids= dict(self.centroids)

            #centroid moving step, for 0 and 1, set new centroid to average of current cluster
            for classification in self.classifications:
                self.centroids[classification]= np.average(self.classifications[classification], axis=0)    #axis=0 averages over all the values

            optimized= True

            for c in self.centroids:
                original_centroid= prev_centroids[c]
                current_centroid= self.centroids[c]
                if np.sum((current_centroid - original_centroid)/original_centroid*100) > self.tol: 
                    optimized= False    #if the centroid moves more than the tolerance then we are not optimized

            if optimized:
                break    #no more iterations if we optimized 

            
    def predict(self, data):
        #to predict for new features, find the closest cluster centroid and assign it
        distances= [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification= distances.index(min(distances))    #assigning the closest cluster
        return classification

#Done writing our own KMeans, time to test it on data


clf= KMeans()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color= "k", s=150, linewidth=5)
    #plots a scatter plot of our 2 centroids
for classification in clf.classifications:
    color= colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],  featureset[1], marker="x", color= color, s=150, linewidth=5)


#add unknown data and use clf.predict(unknown) to cluster assign it
#adding new data here does not change the centroids as the training has already happened
plt.show()

        




























