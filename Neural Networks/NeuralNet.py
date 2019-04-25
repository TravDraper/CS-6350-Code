# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:08:24 2019

@author: drape
"""

import csv
from random import sample
import numpy as np


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

