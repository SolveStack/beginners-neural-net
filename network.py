import numpy as np
import math
import random
from neuron import Neuron, sig
from layer import Layer, sep

class Network:
    def __init__(self, archSpec):
        self.archSpec = archSpec
        self.input = [0]*archSpec[0] #initialize empty input layer array
        self.layers = [Layer(archSpec[i], archSpec[i-1]) for i in range(1, len(archSpec))] #create hidden layers and output layers

    def __str__(self):
        s = 'my architecture has '+ str(len(self.layers)) +' layers with this number of neurons in each: ' + ''.join(map(sep, map(str, self.archSpec)))[:-2]
        return s
        
    def predict(self, input):
        self.input = input #update input layer
        self.layers[0].feedForward(input) #start feed forward with first hidden layer
        for lr in range(1,len(self.layers)):
            self.layers[lr].feedForward(self.layers[lr-1].getNrVals())
        return self.layers[-1].getNrVals()

