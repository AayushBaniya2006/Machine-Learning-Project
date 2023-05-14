#Categorical cross-entropy calculates the loss from the intended target
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

#Calculates the loss
loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])
#All that can be simplified to loss = -math.log(softmax_output[0]), because of maths stuff

print(loss)