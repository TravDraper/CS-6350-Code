# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 14:13:31 2019

@author: drape
"""

import csv
import TreeClasses as tc
from statistics import median, mean
import math as m
import gain_functions as gf
import random as rand

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

testing = 'test_bank.csv'
training = 'train_bank.csv'

def bag_guess(trees, data, labels, outcomes):
    guessVotes = 0
    for tree in trees:
        guess = predict(tree, data, labels)
        if guess == outcomes[0]:
            guessVotes += 1
        else:
            guessVotes -= 1
    if guessVotes >= 0:
        return outcomes[0]
    else:
        return outcomes[1]
    



def main():
    treenums = [1,100,200,300,400,500,700,900,1000]
    trainset, trainraw = makeData(training, labels)
    testset, testraw = makeData(testing, labels)
    medians = medianAssign(trainset, labels)
    trainset = removeNums(trainset, medians)
    testset = removeNums(testset, medians)
    
    trainLabels = [item[-1] for item in trainraw]
    testLabels  = [item[-1] for item in testraw]
    
    for element in trainset:
        element['weight'] = 1
    trainAcc = []
    testAcc = []
    
    for num in treenums:
        treelist = []
        for i in range(num):
            newTraining = rand.choices(trainset,k = len(trainset))
            newTree =  id3(newTraining, labels, label_attr, labels[-1], 18, 'entropy', None)
            treelist.append(newTree)
        trainPred = []
        testPred = []
        for entry in trainset:
            thing = bag_guess(treelist, entry, labels, label_attr['outcome'])
            trainPred.append(thing)
        for entry in testset:
            thing = bag_guess(treelist,entry, labels, label_attr["outcome"])
            testPred.append(thing)
        trainAcc.append(accuracy(trainPred,trainLabels))
        testAcc.append(accuracy(testPred,testLabels))
        print(trainAcc, 'train accuracy')
        print(testAcc, 'test accuracy')

    tree_preds = []
    basics = []
    for i in range(100):
        train_i = rand.choices(trainset,k = 1000)
        treelist_i = []
        for j in range(1000):
            train_j = rand.choices(train_i,k = 1000)
            newTree = id3(train_j, labels, label_attr, labels[-1], 18, 'entropy', None)
            treelist_i.append(newTree)
#            if j%100 == 0:
#                print("100 more trees from set", i, 'have been trained.  iteration = ',j)
        tree_preds.append(treelist_i)
        basics.append(treelist_i[0])
        print("Tree set", i, "has been trained")
    
    
    singleVar = []
    singleBias = []
    singleMean = []
    for entry in testset:
        guess_agg = 0
        predictions = []
        for tree in basics:
            guess = predict(tree, entry, labels)
            if guess == label_attr['outcome'][0]:
                guess_agg += 1
                predictions.append(1)
            else:
                predictions.append(0)
        ave = guess_agg/len(basics)
        singleMean.append(ave)
        value = 0
        if entry['outcome'] == label_attr['outcome'][0]:
            value = 1
        bias = (value - ave)**2
        singleBias.append(bias)
        subVar = []
        for h in predictions:
            mini = (h - ave)**2
            subVar.append(mini)
        var = (1/(len(basics)-1))* sum(subVar)
        singleVar.append(var)
        
    bagVar = []
    bagBias = []
    bagMean = []
    for entry in testset:
        guess_agg = 0
        predictions = []
        for trees in tree_preds:
            guess = bag_guess(trees, entry, labels, label_attr['outcome'])
            if guess == label_attr['outcome'][0]:
                guess_agg += 1
                predictions.append(1)
            else:
                predictions.append(0)
        ave = guess_agg/len(basics)
        bagMean.append(ave)
        value = 0
        if entry['outcome'] == label_attr['outcome'][0]:
            value = 1
        bias = (value - ave)**2
        bagBias.append(bias)
        subVar = []
        for h in predictions:
            mini = (h - ave)**2
            subVar.append(mini)
        var = (1/(len(basics)-1))* sum(subVar)
        bagVar.append(var)

    sVariance = mean(singleVar)
    sBias = mean(singleBias)
    sMSE = sBias + sVariance
    print("The bias and the variance of the single trees are: Variance:", sVariance,
          'Bias:', sBias, "and the general squared error is:", sMSE)
    bVariance = mean(bagVar)
    bBias = mean(bagBias)
    bMSE = bBias + bVariance
    print("The bias and the variance of the bagged trees are: Variance:", bVariance,
          'Bias:', bBias, "and the general squared error is:", bMSE)
    
#main()

def main_slow():
    treenums = [1,100,200,300,400,500,700,900,1000]
    trainset, trainraw = makeData(training, labels)
    testset, testraw = makeData(testing, labels)
    medians = medianAssign(trainset, labels)
    trainset = removeNums(trainset, medians)
    testset = removeNums(testset, medians)
    
    trainLabels = [item[-1] for item in trainraw]
    testLabels  = [item[-1] for item in testraw]
    
    for element in trainset:
        element['weight'] = 1
    trainAcc = []
    testAcc = []
    
    for num in treenums:
        treelist = []
        for i in range(num):
            newTraining = rand.choices(trainset,k = len(trainset))
            newTree =  id3(newTraining, labels, label_attr, labels[-1], 18, 'entropy', None)
            treelist.append(newTree)
        trainPred = []
        testPred = []
        for entry in trainset:
            thing = bag_guess(treelist, entry, labels, label_attr['outcome'])
            trainPred.append(thing)
        for entry in testset:
            thing = bag_guess(treelist,entry, labels, label_attr["outcome"])
            testPred.append(thing)
        trainAcc.append(accuracy(trainPred,trainLabels))
        testAcc.append(accuracy(testPred,testLabels))
        print(trainAcc, 'train accuracy')
        print(testAcc, 'test accuracy')

    tree_preds = []
    basics = []
    for i in range(50):
        train_i = rand.choices(trainset,k = 1000)
        treelist_i = []
        for j in range(300):
            train_j = rand.choices(train_i,k = 1000)
            newTree = id3(train_j, labels, label_attr, labels[-1], 18, 'entropy', None)
            treelist_i.append(newTree)
#            if j%100 == 0:
#                print("100 more trees from set", i, 'have been trained.  iteration = ',j)
        tree_preds.append(treelist_i)
        basics.append(treelist_i[0])
        print("Tree set", i, "has been trained")
    
    
    singleVar = []
    singleBias = []
    singleMean = []
    for entry in testset:
        guess_agg = 0
        predictions = []
        for tree in basics:
            guess = predict(tree, entry, labels)
            if guess == label_attr['outcome'][0]:
                guess_agg += 1
                predictions.append(1)
            else:
                predictions.append(0)
        ave = guess_agg/len(basics)
        singleMean.append(ave)
        value = 0
        if entry['outcome'] == label_attr['outcome'][0]:
            value = 1
        bias = (value - ave)**2
        singleBias.append(bias)
        subVar = []
        for h in predictions:
            mini = (h - ave)**2
            subVar.append(mini)
        var = (1/(len(basics)-1))* sum(subVar)
        singleVar.append(var)
        
    bagVar = []
    bagBias = []
    bagMean = []
    for entry in testset:
        guess_agg = 0
        predictions = []
        for trees in tree_preds:
            guess = bag_guess(trees, entry, labels, label_attr['outcome'])
            if guess == label_attr['outcome'][0]:
                guess_agg += 1
                predictions.append(1)
            else:
                predictions.append(0)
        ave = guess_agg/len(basics)
        bagMean.append(ave)
        value = 0
        if entry['outcome'] == label_attr['outcome'][0]:
            value = 1
        bias = (value - ave)**2
        bagBias.append(bias)
        subVar = []
        for h in predictions:
            mini = (h - ave)**2
            subVar.append(mini)
        var = (1/(len(basics)-1))* sum(subVar)
        bagVar.append(var)

    sVariance = mean(singleVar)
    sBias = mean(singleBias)
    sMSE = sBias + sVariance
    print("The bias and the variance of the single trees are: Variance:", sVariance,
          'Bias:', sBias, "and the general squared error is:", sMSE)
    bVariance = mean(bagVar)
    bBias = mean(bagBias)
    bMSE = bBias + bVariance
    print("The bias and the variance of the bagged trees are: Variance:", bVariance,
          'Bias:', bBias, "and the general squared error is:", bMSE)

