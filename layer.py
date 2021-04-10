from neuron import Neuron, sig


def sep(s):
    return s + ", "


class Layer:
    def __init__(self, layerDim, prevLayerDim):
        self.neurons = [Neuron(prevLayerDim) for i in range(layerDim)]

    def __str__(self):
        s = "my neurons are " + ", ".join(map(str, self.neurons)) + "\n"
        return s

    def getNeuronVals(self):
        return [self.neurons[i].firingVal for i in range(len(self.neurons))]

    def feedForward(self, input):
        for i in range(len(self.neurons)):
            self.neurons[i].deriveFiringVal(input)

    def updateNeurons(self, parameters):
        # [[w], b]...
        for neuron in range(len(parameters)):
            newWeights = parameters[neuron][0]
            newBias = parameters[neuron][1]
            self.neurons[neuron].selfUpdate(newWeights, newBias)

    def randomUpdateNeurons(self, step_size):
        for neuron in range(len(self.neurons)):
            self.neurons[neuron].randomSelfUpdate(step_size)
