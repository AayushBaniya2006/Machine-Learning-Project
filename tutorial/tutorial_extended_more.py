import math
import numpy as np
import nnfs

nnfs.init()

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

#E = math.e

exp_values = np.exp(layer_outputs) #Exponentiation
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#print(np.sum(layer_outputs, axis=1)) #axis determines number of columns


'''
norm_values = exp_values / np.sum(exp_values) #Normalization
print(norm_values)
print(sum(norm_values))

#The combination of exponentiation and normalization creates a softmax activation function




for output in layer_outputs: #Helps remove negative values while not removing the meaning of a negative value - ie exponentiation
    exp_values.append(E**output)

print(exp_values)
norm_base = sum(exp_values)
norm_values = []

for value in exp_values: #normalization
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values)) #Sums up values -> Shows total of normalized values add up to 1
'''