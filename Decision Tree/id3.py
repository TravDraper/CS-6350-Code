# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:35:08 2019

@author: Travis Draper
"""

import math
import TreeClasses as tc
import gain_functions as gf

def id3(data, labels, attr_list, target, depth, gainFunc, path):
    gains, split = gf.best_split(data,labels,target, gainFunc)
    del gains [-1]
    if max(gains) <= 0 or depth <= 1:
        depth -= 1
        prediction = gf.best_guess(data, target)
        return tc.Leaf(split, prediction, path)
    
    else:
        depth = depth - 1
        node = tc.Node(split, depth, path)
        subdata = gf.splitter(data, split)
        i = 0
        #limit = len(attr_list[split])
        for item in attr_list[split]:
            #splitoff = subdata[item]
            if item not in subdata:
            #len(splitoff) == 0:
                prediction = gf.best_guess(data, target)
                #print("We predict:", prediction)
                node.children.append(i)
                node.children[i] = tc.Leaf(item, prediction, item)
                #node.addChild(tc.Leaf(item, prediction))
                i += 1
            else:
                node.children.append(i)
                node.children[i] = id3(subdata[item], labels, attr_list, target, depth, gainFunc, item)
                i += 1
                
        return(node)


def predict(node, data, attributes):
    if isinstance(node,tc.Leaf):
        return node.prediction
    else:
        node_split = node.attribute
        for child in node.children:
            if child.path == data[node_split]:
                label = predict(child, data, attributes)
                return label


def accuracy(guessed, actual):
    if len(guessed) == len(actual):
        size = len(guessed)
        correct = 0
        for i in range(size):
            if guessed[i] == actual[i]:
                correct += 1
        return correct/size
    else:
        print("The sets don't match up")
        return
    

def printTree(node, spacing = ""):
    if isinstance(node,tc.Leaf):
        print(spacing + "Predict " + node.prediction)
        return
    print(spacing + str(node.attribute))
    for child in node.children:
        print(spacing + str(child.path))
        printTree(child, spacing + "  ")
        

