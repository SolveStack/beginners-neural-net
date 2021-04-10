import numpy as np
import math
import random as rd
from neuron import Neuron, sig
from layer import Layer, sep
from network import Network

rd.seed(2021)
testTrainingData = [[[5], [0.2]], [[2], [0.7]], [[0], [0]], [[10], [0.9]], [[4], [0.3]]]
testArchSpec = [1, 3, 3, 5, 1]
testNetwork = Network(testArchSpec)


# print(
# testNetwork.calculateCost(testTrainingData))


# testNetwork.randomParameterUpdate(3)

# print(
# testNetwork.calculateCost(testTrainingData))

testNetwork.train(testTrainingData)

##tLr = Layer(3, 3)
##print(tLr)
##tLr.feedForward([0.1, 0.4, 0.9])
##print(tLr)

# tNw = Network([3, 3, 3, 2])

# tIp = [1, 5, 20]


# print(''.join(map(sep, map(str, tNw.predict(tIp))))[:-2])

# arr = [[1, 2],[2, 3],[3, 4], [4, 5]]

# print(len(arr))
