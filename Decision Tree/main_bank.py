# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:34:08 2019

@author: drape
"""

import csv
import TreeClasses as tc
from statistics import median
import gain_functions as gf

from id3 import id3, predict, accuracy



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

def fixUnknown(data, labels):
    fix = {}
    for attr in labels:
        val_freq = {}
        for record in data:
            if (record[attr] in val_freq):
                val_freq[record[attr]] += 1.0
            else:
                val_freq[record[attr]]  = 1.0
        if "unknown" in val_freq:
            del val_freq["unknown"]
        best = max(val_freq.keys(), key =( lambda k: val_freq[k]))
        fix[attr] = best
        
    return fix


        
def replaceUnknown(data, labels, fix):
    for item in data:
        for attr in labels:
            if item[attr] == 'unknown':
                item[attr] = fix[attr]
    return data


testing = 'test_bank.csv'
training = 'train_bank.csv'
    
#trainset, trainraw = makeData(training, labels)
#testset, testraw = makeData(testing, labels)


#print(trainset[0])



def main():
    trainset, trainraw = makeData(training, labels)
    testset, testraw = makeData(testing, labels)
    medians = medianAssign(trainset, labels)
    trainset = removeNums(trainset, medians)
    testset = removeNums(testset, medians)
    fix = fixUnknown(trainset, labels)
    trainsetU = replaceUnknown(trainset, labels, fix)
    testsetU = replaceUnknown(testset, labels, fix)
    
    trainLabels = [item[-1] for item in trainraw]
    testLabels  = [item[-1] for item in testraw]
    #print(trainset[1])
    print("Running decision tree algorithm on the bank dataset with unknown values")
    algotype = ['gini', 'entropy', 'ME']
    for item in algotype:
        for i in range(1,17):
            currentTree = id3(trainset, labels, label_attr, labels[-1], i, item, None)
            
            trainPred = [predict(currentTree, x, labels) for x in trainset]
            testPred  = [predict(currentTree, x, labels) for x in testset]
            
            trainAcc = accuracy(trainPred, trainLabels)
            testAcc  = accuracy(testPred, testLabels)
            
            print("Decision tree of depth", i, "using", item, "has a test accuracy of", testAcc,
                  'and a training accuracy of', trainAcc)
     
        
    print("Running decision tree algorithm on the bank dataset with unknown's replaced")
    print("\n \n \n \n \n")
    
    for item in algotype:
        for i in range(1,17):
            currentTree = id3(trainsetU, labels, label_attr, labels[-1], i, item, None)
            
            trainPred = [predict(currentTree, x, labels) for x in trainsetU]
            testPred  = [predict(currentTree, x, labels) for x in testsetU]
            
            trainAcc = accuracy(trainPred, trainLabels)
            testAcc  = accuracy(testPred, testLabels)
            
            print("Decision tree of depth", i, "using", item, "has a test accuracy of", testAcc,
                  'and a training accuracy of', trainAcc)
    
main()

