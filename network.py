import numpy as np
import math
import random
from neuron import Neuron, sig
from layer import Layer, sep


class Network:
    def __init__(self, archSpec):
        self.archSpec = archSpec
        self.input = [0] * archSpec[0]  # initialize empty input layer array
        self.layers = [
            Layer(archSpec[i], archSpec[i - 1]) for i in range(1, len(archSpec))
        ]  # create hidden layers and output layer

    def __str__(self):
        s = (
            "my architecture has "
            + str(len(self.layers))
            + " layers with this number of neurons in each: "
            + "".join(map(sep, map(str, self.archSpec)))[:-2]
        )
        s = (
            s
            + "my arch has these neurons: "
            + "".join(map(str, map(str, self.layers)))[:-2]
            + "\n"
        )
        return s

    def feedForward(self, input):
        self.input = input  # update input layer
        self.layers[0].feedForward(input)  # start feed forward with first hidden layer
        for layer in range(1, len(self.layers)):
            self.layers[layer].feedForward(self.layers[layer - 1].getNeuronVals())
        return self.layers[-1].getNeuronVals()

    # calaculate aberage cost based on training data
    # randomly update parameters based on step size input
    # create training function that takes in step size and number of epochs

    def calculateCost(self, trainingData):
        expectedOutput = [trainingData[i][1] for i in range(len(trainingData))]
        actualOutput = [
            self.feedForward(trainingData[point][0])
            for point in range(len(trainingData))
        ]
        # print(', '.join(map(str, expectedOutput)))
        # print(', '.join(map(str, actualOutput)))
        return sum(np.square(np.subtract(expectedOutput, actualOutput))) / len(
            expectedOutput
        )

    def randomParameterUpdate(self, step_size):
        # step_size is the interval between the individual weights and individual biases

        for layer in range(len(self.layers)):
            self.layers[layer].randomUpdateNeurons(step_size)

    def parameterUpdate(self, parameters):
        for layer in range(len(parameters)):
            newParameters = parameters[layer]
            self.layers[layer].updateNeurons(newParameters)

    # def train(self, trainingData, step_size=1, epochs=10):
    #     # wasnt to take the cost, randomly update parameters and rerun data and compare.
    #     # if going down, take new parameters, else redo random update
    #     previousCost = self.calculateCost(trainingData)
    #     previousLayers = self.layers

    #     for epoch in range(epochs):
    #         print(f"Epoch is {epoch}")
    #         count = 0
    #         cost = previousCost
    #         while previousCost - cost <= 0 and count <= 100:
    #             # print(f"cost is {cost}")
    #             # print(f"previousCost is {previousCost}")
    #             self.randomParameterUpdate(step_size)

    #             cost = self.calculateCost(trainingData)
    #             if previousCost - cost >= 0:
    #                 self.layers = previousLayers
    #             count += 1

    #             # print("=======")
    #             # print("After setting new cost and updating layers")
    #             # print(f"cost is {cost}")
    #             # print(f"previousCost is {previousCost}")
    #         previousCost = cost

    #             # print("=======")
    #             # print("RESETTING COST")
    #             # print(f"cost is {cost}")
    #             # print(f"previousCost is {previousCost}")

    #         print(cost)

    def train(self, trainingData, step_size=1, epochs=10):
        # want to take the cost, randomly update parameters and rerun training data and compare. if going down, take new parameters, else redo random update
        previousCost = self.calculateCost(trainingData)
        previousLayers = self.layers
        for epoch in range(epochs):
            count = 0
            cost = previousCost
            while previousCost - cost <= 0 and count <= 10000:
                self.randomParameterUpdate(step_size)
                cost = self.calculateCost(trainingData)
                if previousCost - cost >= 0:
                    self.layers = previousLayers
                count += 1
            previousCost = cost
            print(cost)

    # #takes in training data as a 2 dimensional array: x_i = input, y_i = expected output
    # #calculates error as (network prediction - expected prediction)^2
    # #averages over all training data instances
    # def calculateCost(self, trainingData):
    #     networkOutput = [self.predict(trainingData[i][0]) for i in range(len(trainingData))]
    #     cost = sum((networkOutput - trainingData[1])^2)
    #     return cost/len(trainingData)

    # #each row of parameters has two entries: (1) a weights vector (2) a bias values
    # #should have check that the dimensions of the weights vector is consistent with archSpec
    # def updateParameters(self, parameters):
    #     for layer in range(len(self.layers)):
    #         self.layers[layer].updateNeurons(parameters[layer][0], parameters[layer][1])
