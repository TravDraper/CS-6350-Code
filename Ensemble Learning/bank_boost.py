# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:34:08 2019

@author: drape
"""

import csv
import TreeClasses as tc
from statistics import median
import math as m
import gain_functions as gf

from id3 import id3, predict, accuracy, printTree



labels = ['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration', 
          'campaign', 'pdays', 'previous', 'poutcome', 'outcome']

label_attr = {'age': ['+', '-'],
              'job':["admin.","unknown","unemployed","management",
                     "housemaid","entrepreneur","student", "blue-collar",
                     "self-employed","retired","technician","services"],
              'marital':["married","divorced","single"],
              'education':["unknown","secondary","primary","tertiary"],
              'default':['yes','no'],
              'balance':['+', '-'],
              'housing':['yes','no'], 
              'loan':['yes','no'],
              'contact':["unknown","telephone","cellular"],
              'day':['+', '-'],
              'month':["jan", "feb", "mar", 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', "nov", "dec"],
              'duration':['+', '-'],
              'campaign':['+', '-'],
              'pdays':['+', '-'],
              'previous':['+', '-'],
              'poutcome':["unknown","other","failure","success"],
              'outcome':['yes','no']}

def makeData(file, labels):
    outset = []
    dictset = []
    with open(file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            outset.append(row)
    for entry in outset:
        dictset.append(dict(zip(labels,entry)))
    for key in label_attr:
        if label_attr[key][0] == '+':
            for entry in dictset:
                entry[key] = int(entry[key])
    return dictset, outset

def medianAssign(data, labels):
    medians = {}
    for category in labels:
        vals_list = []
        if label_attr[category][0] == '+':
            for entry in data:
                vals_list.append(entry[category])
            medians[category] = median(vals_list)
    return medians
            
def removeNums(data, medians):
    for entry in data:
        for key in entry:
            if key in medians:
                if entry[key] <= medians[key]:
                    entry[key] = '-'
                else:
                    entry[key] = '+'
    return data



def boost(data, labels, attr_list, target, iterations, answers):
    treelist = []
    for i in range(iterations):
        normalize = 0
        for item in data:
            normalize += item['weight']
        currentTree = id3(data, labels, attr_list, target, 2, 'entropy', None)
#        predictor = []
#        for item in data:
#            predictor.append([predict(currentTree, item, labels), item['weight']])
        
        trainError = 0
#        print('weight =' , data[0]['weight'])
        for item in data:
            if predict(currentTree, item, labels) != item[labels[-1]]:
                trainError += item['weight']/normalize
#        print(trainError)
#        print(normalize)
#        alpha = 1/2*m.log((1-trainError)/trainError)
        alpha = 1/4*m.log((1-trainError)/trainError)
#        print(m.exp(alpha), m.exp(-alpha))
#        print(alpha, " = alpha")
        for item in data:
            if predict(currentTree, item, labels)!= item[labels[-1]]:
                item['weight'] = item['weight']*m.exp(alpha)
#                print(item['weight'])
            else:
                item['weight'] = item['weight']*m.exp(-alpha)
#                print(item['weight'])
        treelist.append({'tree':currentTree, 'alpha': alpha})
        #print(treelist)
    return treelist

def boostGuess(trees, data, labels, outcomes):
    guess = 0
    for tree in trees:
        prediction = predict(tree['tree'], data, labels)
        if prediction == outcomes[0]:
            guess += tree['alpha']
        else:
            guess -= tree['alpha']
    if guess > 0:
        return outcomes[0]
    else:
        return outcomes[1]

testing = 'test_bank.csv'
training = 'train_bank.csv'




#trainset, trainraw = makeData(training, labels)
#medians = medianAssign(trainset, labels)
#trainset = removeNums(trainset, medians)
##testset, testraw = makeData(testing, labels)
#for item in trainset:
#    item['weight']= 1
#printTree(id3(trainset, labels, label_attr, labels[-1], 2,'entropy', None))
#print(trainset[0])



def main():
    trainset, trainraw = makeData(training, labels)
    testset, testraw = makeData(testing, labels)
    medians = medianAssign(trainset, labels)
    trainset = removeNums(trainset, medians)
    testset = removeNums(testset, medians)
    for element in trainset:
        element['weight'] = 1/len(trainset)
    
    trainLabels = [item[-1] for item in trainraw]
    testLabels  = [item[-1] for item in testraw]
    #print(trainset[1])
    
    train = []
    test = []
    for i in [1,2,4,8,16,20,21]:
        for element in trainset:
            element['weight'] = 1/len(trainset)
        #currentTree = id3(trainset, labels, label_attr, labels[-1], i, 'entropy', None)
        treeList = boost(trainset, labels, label_attr, labels[-1], i, trainLabels)
        trainPred = []
        testPred = []
        for entry in trainset:
            trainPred.append(boostGuess(treeList, entry, labels, label_attr['outcome']))
        for entry in testset:
            testPred.append(boostGuess(treeList, entry, labels, label_attr['outcome']))
        
        trainAcc = accuracy(trainPred, trainLabels)
        if trainAcc < 0.3:
            trainAcc = 1-trainAcc
        testAcc  = accuracy(testPred, testLabels)
        if testAcc < 0.3:
            testAcc = 1- testAcc
        train.append(trainAcc)
        test.append(testAcc)
        print("Boosted decision tree with", i, 
              'iterations, has a training accuracy of', trainAcc)
        print("Boosted decision tree with", i, 
              'iterations, has a testing accuracy of', testAcc)
        
    
    
    
    showtree = boost(trainset, labels, label_attr, labels[-1],100, trainLabels)
    stumpTrainAcc = []
    stumpTestAcc = []
    for tree in showtree:
        stumpTrainPred = []
        stumpTestPred = []
        for entry in trainset:
            stumpTrainPred.append(predict(tree['tree'],entry,labels))
        for entry in testset:
            stumpTestPred.append(predict(tree['tree'],entry,labels))
        stumpTrainAcc.append(accuracy(stumpTrainPred,trainLabels))
#        print(stumpTrainAcc[-1])
        stumpTestAcc.append(accuracy(stumpTestPred,testLabels))
    print('Training stumps', stumpTrainAcc)
    print('Testing stumps', stumpTestAcc)
        
def main_slow():
    
    trainset, trainraw = makeData(training, labels)
    testset, testraw = makeData(testing, labels)
    medians = medianAssign(trainset, labels)
    trainset = removeNums(trainset, medians)
    testset = removeNums(testset, medians)
    for element in trainset:
        element['weight'] = 1/len(trainset)
    
    trainLabels = [item[-1] for item in trainraw]
    testLabels  = [item[-1] for item in testraw]
    #print(trainset[1])
    
    train = []
    test = []
    for i in [1,50,200,500,1000]:
        #currentTree = id3(trainset, labels, label_attr, labels[-1], i, 'entropy', None)
        treeList = boost(trainset, labels, label_attr, labels[-1], i, trainLabels)
        trainPred = []
        testPred = []
        for entry in trainset:
            trainPred.append(boostGuess(treeList, entry, labels, label_attr['outcome']))
        for entry in testset:
            testPred.append(boostGuess(treeList, entry, labels, label_attr['outcome']))
        
        trainAcc = accuracy(trainPred, trainLabels)
        if trainAcc < 0.3:
            trainAcc = 1-trainAcc
        testAcc  = accuracy(testPred, testLabels)
        if testAcc < 0.3:
            testAcc = 1- testAcc
        train.append(trainAcc)
        test.append(testAcc)
        print("Boosted decision tree with", i, 
              'iterations, has a training accuracy of', trainAcc)
        print("Boosted decision tree with", i, 
              'iterations, has a testing accuracy of', testAcc)
        
    
    
    
    showtree = boost(trainset, labels, label_attr, labels[-1],1000, trainLabels)
    stumpTrainAcc = []
    stumpTestAcc = []
    for tree in showtree:
        stumpTrainPred = []
        stumpTestPred = []
        for entry in trainset:
            stumpTrainPred.append(predict(tree['tree'],entry,labels))
        for entry in testset:
            stumpTestPred.append(predict(tree['tree'],entry,labels))
        stumpTrainAcc.append(accuracy(stumpTrainPred,trainLabels))
#        print(stumpTrainAcc[-1])
        stumpTestAcc.append(accuracy(stumpTestPred,testLabels))
    print('Training stumps', stumpTrainAcc)
    print('Testing stumps', stumpTestAcc)
  
    
#main()

