import numpy as np
import pandas as pd
import math
import random
#This seemed easier than uploading mnist to GitHub.
from keras.datasets import mnist

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


#This neural network class has a single hidden layer
class NeuralNetwork:
    def __init__(self, x, y, hidden_layer):
        #The input
        self.input = np.zeros(x)
        #The hidden layer has 16 neurons and 784 weights
        self.weights = np.random.rand(self.input.shape[0],hidden_layer)
        #Each of the hidden neurons will have a bias
        self.bias = np.random.rand(hidden_layer)
        self.y = y
        self.output = np.zeros(y)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2




training_set, testing_set = mnist.load_data()
X_train, y_train = training_set
X_test, y_test = testing_set

print(X_train.shape)
print(y_train.shape)
print((training_set[0].shape))

#Create a neural network with 784 inputs, 10 outputs, and a hidden layer with 16 neurons
MNIST = NeuralNetwork(784,10,16)
