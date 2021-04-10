import numpy as np
import random as rd
import math


def sig(x):
    return 1 / (1 + math.exp(-x))


class Neuron:
    def __init__(self, weightDim):
        self.weights = [rd.random() for i in range(weightDim)]
        self.bias = rd.random()
        self.firingVal = rd.random()

    def __str__(self):
        return f"Firing value: {self.firingVal}, Weights: {self.weights}, Bias: {self.bias} \n"

    def deriveFiringVal(self, previousLayer):
        self.firingVal = sig(np.dot(self.weights, previousLayer) + self.bias)

    def selfUpdate(self, newWeights, newBias):
        self.weights = newWeights
        self.bias = newBias

    def randomSelfUpdate(self, step_size):
        # uniform is for the sigmeud
        newWeights = [rd.uniform(-1, 1) for i in range(len(self.weights))]
        biasAdjustment = rd.uniform(-1, 1) * step_size
        self.weights = np.add(self.weights, newWeights)
        self.bias = np.add(self.bias, biasAdjustment)
