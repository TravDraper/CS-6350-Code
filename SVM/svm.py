# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:47:14 2019

@author: drape
"""

import csv
from random import sample
import numpy as np
from scipy.optimize import minimize

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


trainset =  makeData("train_note.csv")
testset =  makeData("test_note.csv")
# Fixing the outputs to align with an SVM
for row in testset:
    if row[-1] == 0:
        row[-1] = -1
for row in trainset:
    if row[-1] == 0:
        row[-1] = -1
#print(trainset[1])
#print(trainset[1][-1])

weights = []
for i in range(len(trainset[0])-1):
    weights.append(0)
weights = np.array(weights)
testset = np.array(testset)
trainset = np.array(trainset)    


def training(data, weights, epochs, C, learning_rate, d = False):
    
    for epoch in range(epochs):
        if d:
            l_r = learning_rate/(1+epoch*learning_rate/d)
        else:
            l_r = learning_rate/(1+epoch)
        newdata = sample(list(data), k = len(data))
        newdata = np.array(newdata)
        for entry in newdata:
            inputs = entry[:-1]
            slack = np.dot(inputs,weights)
            if 1-entry[-1]*slack >= 0:
                weights = (1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
                weights += l_r*C*len(data)*entry[-1]*inputs
            else:
                weights = (1-l_r)*weights 
                weights[-1]*=1/(1-l_r)
    return weights
                
def predict(entry, weight):
    activation = 0
    for i in range(len(entry)-1):
        activation += entry[i]*weight[i]
    if activation >= 0:
        return 1
    else:
        return -1

def error_test(data, weight):
    error = 0
    for entry in data:
        prediction = predict(entry, weight)
        if entry[-1] != prediction:
            error += 1
    error_rate = error/len(data)
    return error_rate

def primal_main():
    for item in [1,10,50,100,300,500,700]:
        w0 = training(trainset, weights, 100, item/873, 0.01, d = 1/3)
        print(w0)
        
        trainerror = error_test(trainset, w0)
        testerror = error_test(testset, w0)
        
        print(trainerror, "Training error with a 'C' value of", item, '/ 873')
        print(testerror, "Testing error with a 'C' value of", item, '/ 873')
        
#    print("Part A is done")
    for item in [1,10,50,100,300,500,700]:
        w0 = training(trainset, weights, 100, item/873, 0.01)
        print(w0)
        
        trainerror = error_test(trainset, w0)
        testerror = error_test(testset, w0)
        
        print(trainerror, "Training error with a 'C' value of", item, '/ 873')
        print(testerror, "Testing error with a 'C' value of", item, '/ 873')
        

primal_main()
    




def dualopt(data, C):
    y_i = trainset[:,-1]    
    constraint = lambda alpha: alpha.dot(y_i)
    bds = [(0,C) for item in range(len(trainset))]
    bds = tuple(bds)
    alphas = np.random.uniform(low = 0.1,high = C, size = (len(trainset),1))
#    xmat = np.zeros((len(trainset),len(trainset)))
#    for xi in range(len(trainset)):
#        for xj in range(len(trainset)):
#            xmat[xi,xj] = trainset[xi,:-1].dot(trainset[xj,:-1])
#    print(xmat[0])
    xmat = trainset[:,:-2].dot(np.transpose(trainset[:,:-2]))
    def dual(alpha, data = trainset):
        y_i = trainset[:,-1].reshape(len(data),1)
#        entries = trainset[:,:-1]
        return 1/2*np.sum(alpha.dot(np.transpose(alpha))*y_i.dot(np.transpose(y_i))*xmat) - np.sum(alpha) 
    cons = ({'type':'eq', 'fun':constraint})
    result = minimize(dual, alphas,  method='SLSQP', bounds=bds, constraints=cons, options = {'maxiter' : 50})
    return result['x']
#guess = dualopt(trainset,100/873)
#print(guess)

for c_val in [100/873,500/873,700/873]:
    guess = dualopt(trainset,c_val)
    dual_weight = np.sum((guess[i]*trainset[i][:-2] for i in range(len(guess))), axis = 0)
    for i in range(len(guess)):
        if guess[i] > 0:
            bias = dual_weight.dot(trainset[i][:-2]) - trainset[i][-1]
            break
    print(np.append(dual_weight, bias))

def kernopt(data, C, gamma):
    kermat = np.zeros((len(trainset),len(trainset)))
    for xi in range(len(trainset)):
        for xj in range(len(trainset)):
            kermat[xi,xj] = np.exp(-(np.linalg.norm(trainset[xi,:-2]-trainset[xj,:-2])**2)/gamma)
    y_i = trainset[:,-1]    
    constraint = lambda alpha: alpha.dot(y_i)
    bds = [(0,C) for item in range(len(trainset))]
    bds = tuple(bds)
    alphas = np.random.uniform(low = 0.1,high = C, size = (len(trainset),1))
    def kernel_dual(alpha, data = trainset):
        y_i = trainset[:,-1].reshape(len(data),1)
        return 1/2*np.sum(alpha.dot(np.transpose(alpha))*y_i.dot(np.transpose(y_i))*kermat) - np.sum(alpha) 
    cons = ({'type':'eq', 'fun':constraint})
    result = minimize(kernel_dual, alphas,  method='SLSQP', bounds=bds, constraints=cons, options = {'maxiter' : 25})
    return result['x']   
    
def kernel_predict(alphas, vectors, bias, gamma, entry):
    sign = np.sum(alphas[i]*vectors[i][-1]*np.exp(-(np.linalg.norm(vectors[i][:-2]-entry[:-2])**2)/gamma) for i in range(len(alphas)))- bias
    if sign >= 0:
        return 1
    else:
        return -1

def kernel_test(alphas, training, gamma, checking):
    a_i = np.empty(shape = 0)
    vectors = np.empty(shape = (0,len(training[0])))
    for i in range(len(alphas)):
        if alphas[i] > 0:
            a_i = np.append(a_i,alphas[i])
            vectors = np.append(vectors, training[i].reshape((1,len(training[0]))), axis = 0)
    print("There are", len(a_i), 'support vectors at this setting')
    bias = np.sum(alphas[i]*vectors[i][-1]*np.exp(-(np.linalg.norm(vectors[i][:-2]-a_i[0])**2)/gamma) for i in range(len(a_i))) - a_i[0]
    error = 0
    for item in checking:
        guess = kernel_predict(a_i, vectors, bias, gamma, item)
        if guess != item[-1]:
            error += 1
    error_rate = error/len(checking)
    return error_rate

alpha_vec = np.empty(shape = (0,len(trainset)))

for c_val in [100/873,500/873,700/873]:
    for gamma in [.01, 0.1, 0.5, 1, 2, 5, 10, 100]:
        guess = kernopt(trainset, c_val, gamma)
        if c_val == 500/873:
            alpha_vec =  np.append(alpha_vec,guess.reshape((1,len(trainset))), axis = 0)
        print("C =", c_val, 'gamma =', gamma)
        print(kernel_test(guess, trainset, gamma, trainset), 'Training error')
        print(kernel_test(guess, trainset, gamma, testset), 'Testing error')

gammas = [.01, 0.1, 0.5, 1, 2, 5, 10, 100]
for i in range(len(alpha_vec)-1):
    samecount = 0
    for j in range(len(alpha_vec[0])):
        if alpha_vec[i][j] > 0 and alpha_vec[i+1][j] > 0:
            samecount += 1
    print('There are', samecount, 'shared support vectors between', gammas[i], 'and', gammas[i+1])

