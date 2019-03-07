# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:28:27 2019

@author: drape
"""
import math
import random as rand

# Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):

    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (record[target_attr] in val_freq):
            val_freq[record[target_attr]] += record["weight"]
        else:
            val_freq[record[target_attr]]  = record["weight"]
    if len(data) == 0:
        return 0
    # Calculate the entropy of the data for the target attribute
    size = sum(val_freq.values())
    for freq in val_freq.values():
        data_entropy += (-freq/size) * math.log(freq/size, 2) 

    return data_entropy

# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
def gain(data, attr, target_attr):

    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (record[attr] in val_freq):
            val_freq[record[attr]] += record["weight"]
        else:
            val_freq[record[attr]]  = record["weight"]

    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        #print('val:', val, 'record[attr]', record[attr])
        data_subset = [record for record in data if record[attr] == val]
        #print(data_subset)
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

    
def best_split(data, labels, target, method):
    gains = []
    
    if method == 'entropy': #Entropy split
        for attribute in labels:
            gains.append(gain(data, attribute, target))
    
    else:
        print("Please choose one of the following methods, 'gini', 'entropy', or 'ME'")
    
    lgains = dict(zip(labels,gains)) #data set with
    del lgains[target] #to remove the target from possible spots to split
    best = max(lgains.keys(), key=(lambda k: lgains[k]))
#    print("Split on", best, lgains[best])
    return gains, best 

def rand_split(data, labels, target, method, k):
    gains = []
    sublabels = rand.sample(labels[0:-1], k = k)
    sublabels.append(labels[-1])
    if method == 'entropy': #Entropy split
        for attribute in sublabels:
            gains.append(gain(data, attribute, target))
    
    else:
        print("Please choose one of the following methods, 'gini', 'entropy', or 'ME'")
    
    lgains = dict(zip(sublabels,gains)) #data set with
    del lgains[target] #to remove the target from possible spots to split
    best = max(lgains.keys(), key=(lambda k: lgains[k]))
#    print("Split on", best, lgains[best])
    return gains, best 

#A function to split a dataset based on one attribute
def splitter(data, attribute):
    #splitlist = []
    splitdict = {}
    labels = {}
    count = 0
    for item in data:
        if item[attribute] not in labels:
            labels[item[attribute]] = count
            splitdict[item[attribute]] = []
            count += 1
#    for label in labels:
#        splitlist.append([])
        
    for item in data:
        splitdict[item[attribute]].append(item)
        
    #for item in data:
    #    splitlist[labels[item[attribute]]].append(item)
    #print(labels)    
    return splitdict


def best_guess(data, target):
    val_freq = {}
    for record in data:
        if record[target] in val_freq:
            val_freq[record[target]] += 1
        else:
            val_freq[record[target]]  = 1.0
        
    best = max(val_freq.keys(), key=(lambda k: val_freq[k]))
    #print(val_freq)
    return best