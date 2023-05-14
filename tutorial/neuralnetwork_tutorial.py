import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
#np.random.seed(0)
'''
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]] #inputs(ie outputs from a hidden layer before the neuron)

X, y = spiral_data(100, 3)
'''
#inputs = [0, 2, -1, 3,3, -2.7, 1.1, 2.2, -100]
#output = []


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases =   np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: #Activation function that sets output from 0 to input
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # -np.max is there to prevent overflow.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)   
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


'''
layer1 = Layer_Dense(2,5) #Creates 1st layer from input data(X)
activation1 = Activation_ReLU() #Takes all the values from neurons, and produces an activation function that is applied to the entire layer
layer1.forward(X)

activation1.forward(layer1.output)
print(activation1.output)
#basic concept behind a ReLU(Rectified linear function)
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
print(output)


Alternative construction of a ReLU
for i in inputs:
    outpur.append(max(0, i))


weights1 = [0.2, 0.8, -0.5, 1] #individual weights for each input(output) given to the neuron
weights2 = [0.5, -0.91, 0.26, -0.5] #Same idea as weights1, just for a differen neuron
weights3 = [-0.26, -0.27, 0.17, 0.87] #Same idea as weights1, just for a different neuron

#Weights adjust the magnitude of an input
weights = [[0.2, 0.8, -0.5, 1],  
           [0.5, -0.91, 0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
#biases offset an input(change its value or smth)
biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],  
           [-0.5, 0.12, -0.33], 
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

weight = [0.2, 0.8, -0.5, 1.0]
bias = 2

#dot product
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)

bias1 = 2
bias2 = 3
bias3 = 0.5


layer_outputs = [] #Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)

output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1, #calculates the output for neuron 1
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2, #calculates the output for neuron 2
         inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3] #calculates the output for neuron 3

print(output)
'''
