# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:08:40 2019

@author: drape
"""


import csv
from random import sample

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

def predict(entry, weight):
    activation = 0
    for i in range(len(entry)-1):
        activation += entry[i]*weight[i]
    if activation >= 0:
        return 1
    else:
        return 0

def error_test(data, weight):
    error = 0
    for entry in data:
        prediction = predict(entry, weight)
        if entry[-1] != prediction:
            error += 1
    error_rate = error/len(data)
    return error_rate


def training(data, weights, epochs, rate):
    for epoch in range(epochs):
        for entry in data:
            prediction = predict(entry, weights)
            if prediction != entry[-1]:
                error = entry[-1] - prediction
                for w in range(len(weights)):
                    weights[w] = weights[w] + rate*error*entry[w]
#        print("The weights are:", weights, "with an error", error_test(data,weights))
        
        if error_test(data, weights) == 0:
            print("Exiting early from the training at epoch:", epoch)
            return weights
    return weights
        
def voting(data, weights, epochs, rate):
    perceptrons = []
    c_m = 0
    for epoch in range(epochs):
        newdata = sample(data, k = len(data))
        for entry in newdata:
            prediction = predict(entry, weights)
            if prediction != entry[-1]:
                error = entry[-1] - prediction
                w_m = weights.copy()
                perceptrons.append([c_m,w_m])
                for w in range(len(weights)):
                    weights[w] = weights[w] + rate*error*entry[w]
                c_m = 1
                
            else:
                c_m += 1
    return perceptrons

def voted(entry, voters):
    prediction = 0
    for voter in voters:
        prediction += voter[0]*2*(predict(entry, voter[1])- 0.5)
    if prediction >= 0:
        return 1
    else:
        return 0

def voted_error(data, voters):
    error = 0
    for entry in data:
        prediction = voted(entry, voters)
        if entry[-1] != prediction:
            error += 1
    error_rate = error/len(data)
    return error_rate



def av_training(data, weights, epochs, rate):
    averaged = weights.copy()
    for epoch in range(epochs):
        for entry in data:
            prediction = predict(entry, weights)
            if prediction != entry[-1]:
                error = entry[-1] - prediction
                for w in range(len(weights)):
                    weights[w] = weights[w] + rate*error*entry[w]
            for a in range(len(averaged)):
                averaged[a] += weights[a]
    return averaged

        
def main():
    trainset = makeData('train_note.csv')
    testset = makeData('test_note.csv')
    
#    print(trainset[0])
    weights = []
    for i in range(len(trainset[0])-1):
        weights.append(0)
    
    new_weights = training(trainset, weights.copy(), 10, .1)
    print("The weights are:", new_weights)
    print("The training error is:", error_test(trainset, new_weights))
    print("The test error is:", error_test(testset, new_weights))
    print(weights)
    print('\n', "Voting Perceptron \n")
    weights_list = voting(trainset, weights.copy(), 10, .1)
#    importantVectors = []
    weights_list_sorted = sorted(weights_list, key =lambda x: x[0], reverse = True)
    for item in weights_list_sorted:
#        if item[0] > 100:
#            importantVectors.append(item)
        print(item)#, 'appears', item[0], 'times')
    print(len(weights_list))
    print("These are the most influential vectors")
#    for item in importantVectors:
#        print(item)
    print("The voted training error is:", voted_error(trainset, weights_list))
    print("The voted testing error is:", voted_error(testset, weights_list))
    
    averaged_weights = av_training(trainset, weights.copy(), 10, .1)
    print("The averaged weights are:", averaged_weights)
    print("The averaged training error is:", error_test(trainset, averaged_weights))
    print("The averaged test error is:", error_test(testset, averaged_weights))
    

main()   