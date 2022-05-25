from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import random
from sklearn.metrics.cluster import adjusted_rand_score

############################## K means class ##############################

class k_means:
   
    def __init__(self,k,filename):
        self.k = k
        self.filename = filename
        #load data we are gonna cluster
        self.grab_data()

    def grab_data(self):
        #grab and store return data from file "filename"
        my_data = np.genfromtxt(self.filename, delimiter=' ', dtype='str')
        [self.m,self.n] = np.shape(my_data)
        #header row is info that we don't need
        my_data=my_data[1:,:]
        #convert to float
        my_data=my_data.astype(float)
        #last entry on each row is a class
        self.labels = my_data[:,-1]
        #the rest are feature vectors
        self.data = my_data[:,0:(self.n-1)]
        #store the ranges of each feature
        self.ranges = np.ptp(self.data,axis=0)
        #determine how much in each category
        self.class_count = Counter(self.labels)
    
    def dist_metric(self,A,B,R):
        #Given two feature vectors A and B
        #and the range of each feature in a vector R
        #return the euclidean distance between the two feature vecs
        return np.linalg.norm(np.divide(A-B,R))

    #given a set of points and some centroids, assign those points to the centroids
    def closest_centroid(self, points, centroids):
        [sample_num, feature_dims] = np.shape(points)
        # now which cluster does each point belong to?
        result = []
        #we want a list of dynamic arrays to store points in each cluster
        for i in range(self.k):
            result.append([])
        #step 2: assign each data point to closest centroid
        for i in range(sample_num):
            #for each feature vector find the closest point
            point = self.data[i,:]
            #for each centroid calculate distance
            dists=[self.dist_metric(point,centroids[j],self.ranges) for j in range(self.k)]
            #argmin return index of minimum value
            cent_result = np.argmin(dists)
            #RESULT: add index of point (point id) to corresponding array storing points in given cluster
            result[cent_result].append(i)
        print(result)
        return result

    #choose k indicies in the range [0,n]
    #proceed with forgy initialisation:
    #initial centroids are randomly selected points
    def init_centroids(self):
        #Choose k random points as the initial centroids.
        init_indx = random.sample(range(0, self.sample_num), self.k)
        print("initial indices  ~ (0,{}): {}".format(self.sample_num,init_indx))
        centroids = self.data[init_indx,:]
        return centroids

    def cluster(self, iternum):
        #step 1: randomly pick k points to be centroids
        [self.sample_num, self.feature_dims] = np.shape(self.data)
        centroids = self.init_centroids()
        
        #clusters is an array of arrays
        #each array stores the indicies of the feature vectors in that cluster
        #i.e cluster1=[1,4] means self.data[1,:] and self.data[4,:] are in cluster
        clusters = self.closest_centroid(self.data, centroids)
        centroids=[]
        for i in range(iternum):
            centroids_old=centroids
            centroids = []
            for cluster in clusters:
                #calculate new centroid
                #first grab points
                point_arr = self.data[cluster,:]
                #new centroid is the mean position of the points in its cluster
                new_centroid = np.mean(point_arr,axis=0)
                centroids.append(new_centroid)
            #reassign the data points to the closest centroid
            clusters = self.closest_centroid(self.data, centroids)
            #show the clusters
            print(clusters)
            #if centroids == centroids_old, we have converged
            if i!=0:
                if np.allclose(centroids,centroids_old):
                    print("---------CONVERGED----------")
                    break

        #get the labels from the cluster
        #And calculate the adjusted rand index
        #To determine how good the clustering is
        guessed_labels=np.zeros(len(self.labels))
        for (index, cluster) in enumerate(clusters):
            print("Cluster {}: {}".format(index,cluster))
            for point in cluster:
                guessed_labels[point] = index
        print("Guessed labels: {}".format(guessed_labels))
        print("Real labels: {}".format(self.labels))
        score=adjusted_rand_score(guessed_labels,self.labels)
        print("Adjusted rand index score: {}".format(score))


############################## Main loop ##############################  

if __name__=="__main__":
    #args: k value, data file
    cluster_bot = k_means(3,'wine-training.txt')
    #args: max number of iterations
    cluster_bot.cluster(100)
