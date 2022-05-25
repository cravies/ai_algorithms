import numpy as np

class KNN:
   
    def __init__(self,k):
        print("Running KNN with k={}".format(k))
        self.k = k

    def grab_data(self,filename):
        #grab and store return data from file "filename"
        my_data = np.genfromtxt(filename, delimiter=' ', dtype='str')
        [m,n] = np.shape(my_data)
        #header row is info that we don't need
        my_data=my_data[1:,:]
        #convert to float
        my_data=my_data.astype(float)
        #last entry on each row is a class
        labels = my_data[:,-1]
        #the rest is a feature vector
        data = my_data[:,0:(n-1)]
        #feature scaling
        return [labels,data]

    def train(self, filename):
        print("Grabbing training data from {}".format(filename))
        [self.labels,self.data] = self.grab_data(filename)

    def dist_metric(self,A,B,R):
        #Given two feature vectors A and B
        #and the range of each feature in a vector R
        #return the euclidean distance between the two feature vecs
        return np.linalg.norm(np.divide(A-B,R))

    def predict(self, point):
        #want to find the k nearest neighbours to the point
        [m,n] = np.shape(self.data)
        distance = np.zeros([m])
        #get ranges for each feature column for distance calculation
        ranges = np.ptp(self.data,axis=0)
        #for every data-point
        for i in range(m):
            feature_vec = self.data[i,:]
            #get euclidean distance between point and feature vector
            distance[i] = self.dist_metric(feature_vec,point,ranges)
        self.distance = distance
        #grab indicies of nearest neighbours
        knn = np.argsort(distance)[:self.k]
        knn_values=[distance[i] for i in knn]
        nearest_labels=[self.labels[i] for i in knn]
        #return most common label among k nearest neighbours
        return np.median(nearest_labels)

    def test(self,filename):
        print("Grabbing testing data from {}".format(filename))
        [test_labels,test_data] = self.grab_data(filename)
        [m,n] = np.shape(test_data)
        total_correct=0
        guessed_labels=[]
        for i in range(m):
            feature_vec = test_data[i,:]
            pred_label = self.predict(feature_vec)
            guessed_labels.append(pred_label)
            if pred_label==test_labels[i]:
                total_correct+=1
        accuracy=total_correct / m
        print("Actual labels: {}".format(test_labels))
        print("Guessed labels: {}".format(guessed_labels))
        print("Difference: {}".format(test_labels - guessed_labels))
        print("Accuracy {}".format(accuracy))

if __name__=="__main__":
    classifier = KNN(3)
    classifier.train("wine-training.txt")
    classifier.test("wine-test.txt")
