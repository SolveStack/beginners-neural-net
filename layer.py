from neuron import Neuron, sig

def sep(s):
    return s+', '

class Layer:
    def __init__(self, layerDim, prevLayerDim):
        self.neurons = [Neuron(prevLayerDim) for i in range(layerDim)]

    def __str__(self):
        s = 'my neurons are ' + ''.join(map(sep, map(str, self.neurons)))[:-2]
        return s

    def getNrVals(self):
        return [self.neurons[i].frVal for i in range(len(self.neurons))]

    def feedForward(self, input):
        for i in range(len(self.neurons)):
            self.neurons[i].deriveFrVal(input)


