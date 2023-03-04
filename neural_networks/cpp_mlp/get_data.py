import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist
from random import sample
import pandas as pd

(x_train, y_train), (x_test, y_test) = mnist.load_data()
SCALE_FACTOR = 255
WIDTH = x_train.shape[1]
HEIGHT = x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
x_test = x_test.reshape(x_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR

np.savetxt("x_train.csv", x_train, delimiter=",", fmt = '%.4f')
np.savetxt("x_test.csv", x_test, delimiter=",", fmt = '%.4f')
np.savetxt("y_train.csv", y_train, delimiter=",", fmt = '%.4f')
np.savetxt("y_test.csv", y_test, delimiter=",", fmt = '%.4f')
