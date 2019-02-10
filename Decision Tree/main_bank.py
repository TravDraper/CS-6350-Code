# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 23:34:08 2019

@author: drape
"""

import csv
import TreeClasses as tc

import gain_functions_numeric as gfn



labels = ['age', 'job', 'marital', 'education', 'default', 'balance',
          'housing', 'loan', 'contact', 'day', 'month', 'duration', 
          'campaign', 'pdays', 'previous', 'poutcome', 'outcome']

label_attr = {'age': 'numeric',
              'job':["admin.","unknown","unemployed","management",
                     "housemaid","entrepreneur","student", "blue-collar",
                     "self-employed","retired","technician","services"],
              'marital':["married","divorced","single"],
              'education':["unknown","secondary","primary","tertiary"],
              'default':['yes','no'],
              'balance':'numeric',
              'housing':['yes','no'], 
              'loan':['yes','no'],
              'contact':["unknown","telephone","cellular"],
              'day':'numeric',
              'month':["jan", "feb", "mar", 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', "nov", "dec"],
              'duration':'numeric',
              'campaign':'numeric',
              'pdays':'numeric',
              'previous':'numeric',
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
    
    return dictset, outset


    
    