import numpy as np
import math
import random
from neuron import Neuron, sig
from layer import Layer, sep
from network import Network

##tLr = Layer(3, 3)
##print(tLr)
##tLr.feedForward([0.1, 0.4, 0.9])
##print(tLr)

tNw = Network([3, 3, 3, 2])

tIp = [1, 5, 20]


print(''.join(map(sep, map(str, tNw.predict(tIp))))[:-2])
