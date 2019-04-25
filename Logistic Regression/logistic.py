# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:27:57 2019

@author: drape
"""

import csv
from random import sample
import numpy as np


def makeData(file):
    inset = []
    with open(file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            inset.append(row)
    matrix = []
    for row in inset:
        # Inserting the Bias term into the elements
        row.insert(-1,1)
        matrix.append([float(i) for i in row])
    return matrix

def predict(entry, weights):
    weights = np.array(weights)
    if len(entry) > len(weights):
        entry = np.array(entry[:-1])
    else:
        entry = np.array(entry)
    yhat = entry.dot(weights)
    return sigmoid(yhat)
    
testset = makeData('test_note.csv')
trainset = makeData('train_note.csv')

def sigmoid(expression):
    output =  1/(1 + np.exp(-1*expression))
    return output

def mapWeights(training, initial, rate_0, d_val, sigma2, epochs):
    weight =  np.array(initial)
    
    for epoch in range(epochs):
        t = epoch + 1
        l_r = rate_0/(1 + (rate_0/d_val)*t)
#        print('Learning Rate',l_r)
        newdata = sample(training, k = len(training))
        newdata = np.array(newdata)
        for entry in newdata:
            x_i = entry[:-1]
            if entry[-1] == 1:
                y_i = 1
            else:
                y_i = -1
#            print(l_r*(len(training)*y_i*x_i*(1-sigmoid(y_i*weight.dot(x_i))) - 1/sigma2*weight))
#            print(l_r*(len(training)*y_i*x_i*(1-sigmoid(y_i*weight.dot(x_i))) - 1/sigma2*weight))
            weight = weight + l_r*(len(training)*y_i*x_i*(1-sigmoid(y_i*weight.dot(x_i))) - 1/sigma2*weight)
    return weight

def mleWeights(training, initial, rate_0, d_val, epochs):
    weight =  np.array(initial)
    for epoch in range(epochs):
        t = epoch + 1
        l_r = rate_0/(1 + (rate_0/d_val)*t)
        newdata = sample(training, k = len(training))
        newdata = np.array(newdata)
        for entry in newdata:
            yhat = predict(entry, weight)
            error = entry[-1] - yhat
            weight = weight + l_r*error*yhat*(1-yhat)*entry[:-1]
    return weight


def accuracy(weights, data):
    errors = 0
    for entry in data:
        out = predict(entry, weights)
        if out > 0.5:
            out = 1
        else:
            out = 0
        if out != entry[-1]:
            errors += 1
    return errors/len(data)



for sigma in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    weights = np.random.normal(0,sigma,(len(trainset[0])-1))
    t_weights =mapWeights(trainset, weights, 0.001, 1, sigma, 50)
    
    print('Training',1 - accuracy(t_weights, trainset), 'Testing', 1 - accuracy(t_weights, testset))
   
#for d in [.01, .1, .5, 1, 10]:
#    for rate in [.001, .01, .05, .1, 0.5]:
#        likely_weights = mleWeights(trainset, weights, rate, d, 10)
#        print('d value:', d, 'learning rate:', rate)
#        print('Training',1 - accuracy(likely_weights, trainset), 'Testing', 1 - accuracy(likely_weights, testset))
        
        
        
for sigma in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    n_weights = np.random.normal(0,sigma,(len(trainset[0])-1))
    likely_weights = mleWeights(trainset, n_weights, 0.5, 0.5,50)
    print("Sigma Value", sigma, '\n', 'Training',1 - accuracy(likely_weights, trainset), 'Testing', 1 - accuracy(likely_weights, testset))