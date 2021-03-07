import random as rd
import numpy as np
from math import exp


def sep(s):
    return s + ", "


class Neuron:
    def __init__(self, numWeights) -> None:
        self.weights = []
        self.bias = []
        self.value = rd.random()
        for i in range(0, numWeights):
            self.weights.append(rd.random())
            self.bias.append(rd.random())

    def __str__(self):
        return str(self.value)

    def setVal(self, value):
        self.value = value

    def findVal(self, prevLayer):
        np.dot(self.weights, prevLayer) + self.bias


class Layer:
    def __init__(self, numWeights, numNeurons) -> None:
        self.neurons = []
        for i in range(0, numNeurons):
            self.neurons.append(Neuron(numWeights))

    def __str__(self):
        s = (
            "my neurons are " + "".join(map(sep, map(str, self.neurons)))[:-2]
        )  # [.1, .35, .12]
        return s


# archSpec: [ 1, 4, 5, ]
class Network:
    def __init__(self, archSpec) -> None:
        self.archSpec = archSpec
        self.layers = []
        for i in range(0, len(archSpec)):
            numWeights = archSpec[i - 1]
            numNeurons = archSpec[i]
            self.layers.append(Layer(numWeights, numNeurons))

    def __str__(self):
        s = (
            "my architecture is: "
            + str(len(self.archSpec))
            + " layers with this number of neurons in each: "
            + "".join(map(sep, map(str, self.archSpec)))[:-2]
        )
        return s


tNr = Neuron(5)
tLr = Layer(3, 5)
tNet = Network([1, 5, 9])

print(tNr)
print(tLr)
print(tNet)
