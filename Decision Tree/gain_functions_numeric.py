# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:28:27 2019

@author: drape
"""
import math


# Calculates the entropy of the given data set for the target attribute.
def entropy(data, target_attr):

    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        if (record[target_attr] in val_freq):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0
    if len(data) == 0:
        return 0
    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq/len(data)) * math.log(freq/len(data), 2) 

    return data_entropy

# Calculates the information gain (reduction in entropy) that would result by splitting the data on the chosen attribute (attr).
def gain(data, attr, target_attr):

    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        if (record[attr] in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0

    # Calculate the sum of the entropy for each subset of records weighted by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        #print('val:', val, 'record[attr]', record[attr])
        data_subset = [record for record in data if record[attr] == val]
        #print(data_subset)
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)

def gini(data, target_attr):
    val_freq = {}
    score = 0
    #find the frequency of the target value
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]]  = 1.0
    total = sum(val_freq.values())
    if total == 0:
        #Check to see if we actually have any items, if not, we return 0
        return 0
    
    for freq in val_freq.values():
        score += (freq/len(data))**2
        
    impurity = 1 - score
    return impurity
            
def gainG(data, attr, target_attr):
    val_freq = {}
    gindex = 0.0
    #find the frequency of the target value
    
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0
    
    for val in val_freq.keys():
        #print("Val is:", val)
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = []
        for record in data:
            #print("record[attr] is:", record[attr])
            if record[attr] == val:
                #print(record)
                data_subset.append(record)
                
        #print("Partial:", val_prob * gini(data_subset, target_attr))
        gindex += val_prob * gini(data_subset, target_attr)
    #print(gini(data, target_attr))    
    return(gini(data, target_attr) - gindex)    
    
def majorityError(data, target_attr):
    val_freq = {}
    error = 0.0
    #find the frequency of the target values
    for record in data:
        if record[target_attr] in val_freq:
            val_freq[record[target_attr]] += 1
        else:
            val_freq[record[target_attr]]  = 1.0
    #calculate majority error
    top_val = max(val_freq.values())
    total = sum(val_freq.values())
    error = (total-top_val)/total
    return error    

def gainE(data, attr, target_attr):

    val_freq = {}
    subset_error = 0.0
    
    for record in data:
        if (record[attr] in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]]  = 1.0
            
    #Calculate the ME values for each subset of records
    
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        
        subset_error += val_prob * majorityError(data_subset, target_attr)
        print("Probability:", val_prob, "ME:",  majorityError(data_subset, target_attr))    
    #subtract ME of chosen attribute from ME of whole set wrt target attribute
    print(majorityError(data, target_attr))
    return(majorityError(data, target_attr) - subset_error)
    
    
def best_split(data, labels, target, method):
    gains = []
    if method == 'gini': #Gini split
        for attribute in labels:
            gains.append(gainG(data, attribute, target))
    elif method == 'entropy': #Entropy split
        for attribute in labels:
            gains.append(gain(data, attribute, target))
    elif method == 'ME': #majority error split
        for attribute in labels:
            gains.append(gainE(data, attribute, target))
    else:
        print("Please choose one of the following methods, 'gini', 'entropy', or 'ME'")
    
    lgains = dict(zip(labels,gains)) #data set with
    del lgains[target] #to remove the 
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