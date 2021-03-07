import numpy as np;
import random as rd;
import math;

def sig(x):
    return 1/(1+math.exp(-x))

class Neuron:
    def __init__(self, weightDim):
        self.weights = [rd.random() for i in range(weightDim)]
        self.bias = rd.random()
        self.frVal = rd.random()

    def __str__(self):
        return str(self.frVal)
    
    def deriveFrVal(self, input):
        self.frVal = sig(np.dot(self.weights, input) + self.bias)
    
        
        
