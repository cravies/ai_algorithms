import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class perceptron:

    def __init__(self,filename,class_true,class_false,eta):
        self.class_true=class_true
        self.class_false=class_false
        self.eta  = eta
        self.bias = 1
        [self.data,self.labels] = self.grab_data(filename)
        self.labels=self.labels.astype(int)
        #if we have a testing, training split, split the data
        [m,n] = np.shape(self.data)
        #random weights
        self.weights = 0.5*np.ones(n)
        #test and train data.. to test would set test data to 
        #different dataset, didn't implement this.
        self.train_data = self.data[int(0.1*m):,:]
        self.test_data = self.data[0:int(0.1*m),:]
        self.train(self.train_data,5000)

    def grab_data(self,filename):
        #grab and store return data from file "filename"
        #read in with pandas
        my_data = np.loadtxt(filename, delimiter=',', dtype='str')
        my_data = shuffle(my_data)
        print(f"Loading dataset... {my_data}")
        #split into data and labels
        [m,n] = np.shape(my_data)
        #print("shape of pandas dataframe is {},{}".format(m,n))
        [data,labels] = np.split(my_data, [(n-1)], axis=1)
        #convert class labels to boolean
        labels=np.char.replace(labels,self.class_true,'1')
        labels=np.char.replace(labels,self.class_false,'0')
        return [data.astype(float),labels.astype(int)]

    def f(self,x):
        """define our activation function for our layer
            here we use threshold function """
        if x > 0:
            return 1
        else:
            return 0

    def train(self,train_data,epochs):
        """ Train the perceptron for a set number of epochs
            or until convergence """
        total_correct=0
        total_updates=0
        epoch=0
        acc=0
        best_acc=0
        best_bias = None
        best_weights = None
        best_epoch=None
        self.train_data = train_data
        print(f"train data {train_data}")
        [data_size, feature_num] = np.shape(self.train_data)
        best_weights = self.weights
        best_bias = self.bias
        best_acc = 0
        for epoch in range(epochs):
            print(f"bias {self.bias}")
            print(f"weights {self.weights}")
            print(f"-----EPOCH {epoch}/{epochs}------")
            old_weights = self.weights
            old_bias = self.bias
            weights = np.zeros(np.shape(self.weights))
            bias = 0
            for i in range(data_size):
                example = self.train_data[i,:]
                activation = self.f(np.dot(self.weights,example)+self.bias)
                real_activation = self.labels[i][0]
                #print("[guess,real]={}".format([activation,real_activation]))
                if activation == real_activation:
                    total_correct += 1
                elif activation != real_activation:
                    #define variables
                    if real_activation == 1:
                        #high activation, we are too low
                        bias += self.eta
                        weights += self.eta * example
                    else:
                        #low activation, we are too high
                        bias -= self.eta
                        weights -= self.eta * example
            #now we apply all the changes at once.
            self.weights += weights
            self.bias += bias
            total_updates += 1
            acc = self.test(self.test_data)
            if acc > best_acc:
                best_epoch = epoch
                best_acc = acc
                best_weights = self.weights
                best_bias = self.bias
        #revert to best weights and bias
        print(f"best accuracy {best_acc} at epoch {best_epoch}")
        self.weights = best_weights
        self.bias = best_bias
        print("Did {} training iterations.".format(total_updates))
        print("Did {} epochs.".format(epoch))
        print("Bias {}".format(self.bias))
        print("Weights {}".format(self.weights))

    def test(self,test_data):
        """ Test the trained perceptron. """
        print("------------TESTING------------")
        #test the perceptron now that it has been trained.
        total_correct=0
        self.test_data = test_data
        [data_size, feature_num] = np.shape(self.test_data)

        #start testing
        for i in range(data_size):
            #grab row slice, this is an instance
            example = self.test_data[i,:]
            #print(f"example: {example}")
            activation = self.f(np.dot(self.weights,example) + self.bias)
            real_activation = self.labels[i] 
            #print(f"guess {activation} real {real_activation}")
            if activation==real_activation:
                total_correct += 1
        accuracy = (total_correct / data_size)
        print(f"testing accuracy {total_correct}/{data_size} = {accuracy}")
        return accuracy 

if __name__=="__main__":
    #part 1
    print("------------ PART 1: IONOSPHERE -----------")
    my_nn = perceptron("ionosphere.data",'g','b',0.2)
    print("Finished ionosphere analysis")
    print("-------------------------------------------")
    """
    #part 2
    print("------------ PART 2: TABLE ----------------")
    print("Doing table analysis")
    my_nn = perceptron("table.data",'1','0',0.1)
    accs = []
    for i in range(5):
        acc = my_nn.test()
        accs.append(acc)
    print(f"Mean accuracy over 5 runs: {np.mean(accs)}")
    """
