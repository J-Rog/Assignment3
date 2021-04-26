import numpy as np
import pandas as pd
import math
import random
#This seemed easier than uploading mnist to GitHub.
from keras.datasets import mnist

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# This neural network class has a single hidden layer
class NeuralNetwork:
    def __init__(self, shape):
        self.input = np.zeros(shape[0])
        self.input_bias = np.random.uniform(-1,1,shape[0])
        self.input_weights = np.random.uniform(-1,1,(shape[0],shape[1]))

        self.hidden_layer = np.zeros(shape[1])
        self.hidden_layer_bias = np.random.uniform(-1,1,shape[1])
        self.hidden_layer_weights = np.random.uniform(-1,1,(shape[1],shape[2]))

        self.output = np.zeros(shape[2])
        self.output_bias = np.random.uniform(-1,1,shape[2])

        self.cost_function = 0

    def getInput(self, datum):
        for i in range(0,datum.shape[0]):
            for j in range(0,datum.shape[1]):
                self.input[i] = datum[i][j]

    def debug(self):
        print(self.input.shape)

    def feedforward(self):

        total = 0

        for j in range(0,len(self.hidden_layer)):
            for i in range(0,len(self.input)):
                total = self.input[i]*self.input_weights[i][j]
            self.hidden_layer[j] = sigmoid(total + self.hidden_layer_bias[j])

        for j in range(0,len(self.output)):
            for i in range(0,len(self.hidden_layer)):
                total = self.output[j]*self.hidden_layer_weights[i][j]
            self.output[j] = sigmoid(total + self.output_bias[j])

    def setCostFunction(self,answer):

        total = 0

        for i in range(len(self.output)):
            if(i != answer):
                total += self.output[i]**2
            else:
                total+= (self.output[i] - 1)**2

        self.cost_function = total


    def backprop():
        total = 0



training_set, testing_set = mnist.load_data()
X_train, y_train = training_set
X_test, y_test = testing_set

print(X_train[0].shape)
print(y_train.shape)
print((training_set[0].shape))

#Create a neural network with 784 inputs, 10 outputs, and a hidden layer with 16 neurons
MNIST = NeuralNetwork((784,28,10))
MNIST.debug()

#Train the neural network
for i in range(4):
    MNIST.getInput(X_train[i])
    MNIST.feedforward()
