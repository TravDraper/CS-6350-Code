# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:07:58 2019

@author: drape
"""

import tensorflow as tf
from tensorflow import keras

import numpy as np
import csv

def makeData(file):
    inset = []
    with open(file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            inset.append(row)
    matrix = []
    for row in inset:
        # Inserting the Bias term into the elements
#        row.insert(-1,1)
        matrix.append([float(i) for i in row])
        
    return matrix

testset = makeData('test_note.csv')
trainset = makeData('train_note.csv')

traindata = []
trainlabels = []
for entry in trainset:
    traindata.append(entry[:-1])
    trainlabels.append(entry[-1])
    
testdata = []
testlabels = []
for entry in testset:
    testdata.append(entry[:-1])
    testlabels.append(entry[-1])
    
traindata = np.array(traindata)
trainlabels = np.array(trainlabels)
testdata = np.array(testdata)
testlabels = np.array(testlabels)

print(np.shape(traindata))
print(traindata[73])

models = []
for depth in [3,5,9]:
    for width in [5,10,25,50,100]:
        
        model1 = keras.Sequential([
                keras.layers.Dense(4, kernel_initializer='he_normal',
                        bias_initializer='he_normal', activation=tf.nn.relu),    
                ])
        for k in range(depth-1):
            model1.add(keras.layers.Dense(width, kernel_initializer='he_normal',
                        bias_initializer='he_normal',activation=tf.nn.relu))
                
        model1.add(keras.layers.Dense(2, kernel_initializer='he_normal',
                        bias_initializer='he_normal',activation=tf.nn.relu))
        model1.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
        
        model1.fit(traindata, trainlabels, epochs=25)
        models.append(model1)
trainAcc = []
testAcc = []
for model in models:
        
    test_loss, test_accuracy = model.evaluate(testdata, testlabels)
    train_loss, train_accuracy = model.evaluate(traindata, trainlabels)
    trainAcc.append(train_accuracy)
    testAcc.append(test_accuracy)
    
models2 = []
for depth in [3,5,9]:
    for width in [5,10,25,50,100]:
        
        model1 = keras.Sequential([
                keras.layers.Dense(4, kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal', activation=tf.nn.tanh),    
                ])
        for k in range(depth-1):
            model1.add(keras.layers.Dense(width, kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal',activation=tf.nn.tanh))
                
        model1.add(keras.layers.Dense(2, kernel_initializer='glorot_normal',
                        bias_initializer='glorot_normal',activation=tf.nn.tanh))
        model1.compile(optimizer = 'adam', 
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
        
        model1.fit(traindata, trainlabels, epochs=25)
        models2.append(model1)
trainAcc2 = []
testAcc2 = []
for model in models2:
        
    test_loss, test_accuracy = model.evaluate(testdata, testlabels)
    train_loss, train_accuracy = model.evaluate(traindata, trainlabels)
    trainAcc.append(train_accuracy)
    testAcc.append(test_accuracy)

#    print("Test Accuracy:", test_accuracy)
print('Training and Test accuracy using Relu  Activation, and he initializer:\n')
for i in range(len(testAcc)):
    print('Train Accuracy:', trainAcc[i])
    print('Test Accuracy:', testAcc[i])

print('Training and Test accuracy using tanh  Activation, and Xavier (glorot normal) initializer:\n')
for i in range(len(testAcc2)):
    print('Train Accuracy (tanh):', trainAcc2[i])
    print('Test Accuracy (tanh):', testAcc2[i])

