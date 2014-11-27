# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 20:41:09 2014

@author: DASLab
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# first create a spatial seperator with which to create a training set.
# create a line
def create_separator_line():
    gradient = rnd.uniform(-1., 1.) * 5
    offset = rnd.uniform(-1, 1)
    return gradient, offset
    
gradient, offset = create_separator_line()
print "gradient = ", gradient, ", offset = ", offset

def create_data_points(nPoints, gradient, offset):
    dataSet = []
    for iPoint in xrange(nPoints):    
        x1 = rnd.uniform(-1, 1)
        x2 = rnd.uniform(-1, 1)
        if x2 > gradient*x1 + offset:
            y = 1
        else:
            y = -1
        dataSet.append((x1, x2, y))
    return dataSet
    
dataSet = create_data_points(300, gradient, offset)
dataArray = np.array(dataSet)

fig = plt.figure()
plt.scatter(dataArray[:,0], dataArray[:,1], 
            c =  dataArray[:,2])
plt.plot(dataArray[:,0], gradient*dataArray[:,0] + offset, color = "black")
plt.axis([-1, 1, -1, 1])

### Now we need to create a perceptron algorithm and test it.
### This should re-create the separator line created by the create_line_separator
### function

def perceptron(trainingSet, nIterations, learningRate, weightsInitial):
    weights = weightsInitial    
    for iIteration in xrange(nIterations):
        for x1, x2, y in trainingSet:   
            parameters =  [1, x1, x2]   
            parameterWeightProduct = sum(parameter * weight for parameter, 
                                         weight in zip(weights, parameters))
            if parameterWeightProduct*y <= 0:
                weights[0] = weights[0] + y*parameters[0]*learningRate
                weights[1] = weights[1] + y*parameters[1]*learningRate
                weights[2] = weights[2] + y*parameters[2]*learningRate
            #print "current weights: ", weights
    return weights

weightsInitial = [0.5, 0.5, 0.5] # initialise the weights for the algorithm
trainingSet = dataArray ## the training set is the data created above.
nIterations = 100
learningRate = 0.1

weights = perceptron(trainingSet, nIterations, learningRate, weightsInitial)
    
print "final weights: ", weights
print "gradient = ", gradient, ", offset = ", offset

gradientPredicted = -weights[1]/weights[2]
print "predicted gradient = ", gradientPredicted
offsetPredicted = -weights[0]/weights[2]
print "predicted offset = ", offsetPredicted

plt.plot(dataArray[:,0], gradientPredicted*dataArray[:,0] + offsetPredicted,
         color = "green")

#accuracy = sum()
