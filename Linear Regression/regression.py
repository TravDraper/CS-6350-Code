# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 14:55:44 2019

@author: drape
"""

import csv




def makeData(file):
    outset = []
    with open(file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            outset.append(row)
    for row in outset:
        for entry in row:
            entry = float(entry)
    targets = []
    for row in outset:
        targets.append(row[-1])
        # This section separates the output variable, and includes the bias term in the x values.
        row[-1] = 1
    return targets, outset

def gradient(weights, data, output):
    grad = []
    
    for i in range(len(weights)):
        sigma = 0
        for vec in range(len(data)):
            sigma += (output[vec]-sum(p*q for p,q in zip(weights,data[vec])))*data[vec][i]
        grad.append(sigma*-1)
    
    return grad

def errorFunc(output, weights, data):
    totalError = 0
    for i in range(len(data)):
        totalError += (output[i]- sum(p*q for p,q in zip(weights,data[i])))**2
    return totalError
    

def gradDesc(data, outputs, weights, rate, limit):
    costFunc = []
    firstError = errorFunc(outputs, weights, data)
    costFunc.append(firstError)
    grad = gradient(weights, data, outputs)
    for i in range(len(weights)):
        weights[i] = weights[i] - rate*grad[i]
    error = errorFunc(outputs, weights, data)
    costFunc.append(error)
    delta = firstError - error
    
    while delta > limit:
        grad = gradient(weights, data, outputs)
        for i in range(len(weights)):
            weights[i] = weights[i] - rate*grad[i]
        costFunc.append(errorFunc(outputs, weights, data))
        delta = costFunc[-2] - costFunc[-1]
        if len(costFunc) > 8000:
            break
        
    return weights, costFunc

def stochastic(data, output, weights):
    grad = []
    for i in range(len(weights)):
        grad.append(-1*(output - sum(p*q for p,q in zip(weights,data)))*data[i])    
    return grad

def stochasticDesc(data, outputs, weights, rate, limit):
    costFunc = []
    error1 = errorFunc(outputs, weights, data)
    costFunc.append(error1)
    i = 0
    grad = stochastic(data[i],outputs[i],weights)
    for j in range(len(weights)):
        weights[j] = weights[j] - rate*grad[j]
    error2 = errorFunc(outputs, weights, data)
    costFunc.append(error2)
    delta = error1 - error2
    
    while abs(delta) > limit:
        i += 1
        k = len(data)
        grad = stochastic(data[i%k],outputs[i%k],weights)
        for j in range(len(weights)):
            weights[j] = weights[j] - rate*grad[j]
        costFunc.append(errorFunc(outputs, weights, data))
        delta = costFunc[-2] - costFunc[-1]
        if len(costFunc) > 8000:
            break
        
    return weights, costFunc
        

def main():
    target, data = makeData('concrete_train.csv')
    test_y, test_data = makeData('concrete_test.csv')

    matrix = []
    for row in data:
        matrix.append([float(i) for i in row])
    vector = [float(i) for i in target]
    
    test = []
    for row in test_data:
        test.append([float(i) for i in row])
    testvec = [float(i) for i in test_y]
    
    #print(matrix[1])
    #print(vector)
    
    #Let's create our weights vector to start out with
    weights = []
    for i in range(len(matrix[0])):
        weights.append(0)
    
    weight_final, cost = gradDesc(matrix, vector, weights, 0.01, 1e-6)
    print(weight_final)
    print("Gradient at final Weight", gradient(weight_final, matrix, vector))
    print('\n\n\n')
    print(len(cost))
    if len(cost) < 100:
        print(cost)
    print(cost[-1])
    
    print(errorFunc(testvec, weight_final, test))
    print('\n\n\n')
    weights = []
    for i in range(len(matrix[0])):
        weights.append(0)
    
    weight_final, cost = stochasticDesc(matrix, vector, weights, 0.001, 1e-7)
    print(weight_final)
    print("Gradient at final Weight", gradient(weight_final, matrix, vector))
    print('\n\n\n')
    print(len(cost))
    if len(cost) < 100:
        print(cost)
    print(cost[-1])
    
    print(errorFunc(testvec, weight_final, test))
    
    print('\n\n\n')
#    print(cost)
#main()



    
    
    