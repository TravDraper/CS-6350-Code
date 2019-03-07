# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:48:15 2019

@author: drape
"""
import gain_functions as gf


    


def predict(data,target):
    count = {}
    for item in data:
        label = item[target]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count


class Leaf:
    def __init__(self,attribute, prediction, path):
        self.prediction = prediction
        self.attribute = attribute
        self.children = []
        self.path = path
        
class Node:
    def __init__(self, attribute, depth, path):
        self.children = []
        self.depth = depth
        self.attribute = attribute
        self.path = path
        
    def addChild(self, obj):
        self.children.append(obj)
    
class Question:
    def __init__(self, attribute, value):
        self.attribute =  attribute
        self.value = value
        
    
        
    
        
    
        

