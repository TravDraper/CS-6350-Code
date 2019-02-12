# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:05:32 2019

@author: drape
"""
import csv
import TreeClasses as tc
import id3
import gain_functions as gf


labels = ['buying','maint','doors','persons','lug_boot','safety','label']

#The labels that each of the attributes can have
label_attr = {'buying': ['vhigh', 'high', 'med', 'low'],
              'maint': ['vhigh', 'high', 'med', 'low'],
              'doors': ['2','3','4','5more'],
              'persons': ['2','4','more'],
              'lug_boot': ['small', 'med', 'big'],
              'safety': ['low', 'med', 'high'],
              'label': ['unacc', 'acc', 'good', 'vgood'] }
 
training = 'train_car.csv'
testing = 'test_car.csv'



#This segment is for testing purposes
#train_set = []
#with open('train_car.csv') as csvDataFile:
#    csvReader = csv.reader(csvDataFile)
#    for row in csvReader:
#        train_set.append(row)
#        #print(row)
#     
#       
##print(train_set[0])        
#dataset = []
#
#for entry in train_set:
#    dataset.append(dict(zip(labels,entry)))
    
    
    
    
    
    
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


#print(dataset[1])


#gains, split = gf.best_split(dataset, labels, labels[-1], 'entropy')
#print(gains)


#first = gf.splitter(dataset,split)
#print(gf.best_guess(first['high'],labels[-1]))

cardict, carlist = makeData(training, labels)
print(gf.best_split(cardict, labels, labels[-1], "ME"))
print(cardict[1])



        





def main():
    trainset , trainRaw = makeData(training, labels)
    testset, testRaw = makeData(testing, labels)
    trainLabels = [item[-1] for item in trainRaw]
    testLabels  = [item[-1] for item in testRaw]
    
    
    #print(gf.gainE(trainset,labels[5], labels[-1]))
    
#    mytree = id3(trainset, labels, label_attr, labels[-1], 6, "entropy", None)
#    printTree(mytree)
    print("Running the decision tree algorithm on the 'Cars' dataset.")

    algotype = ['gini', 'entropy', 'ME']
    for item in algotype:
        for i in range(1,7):
            currentTree = id3.id3(trainset, labels, label_attr, labels[-1], i, item, None)
            
            trainPred = [id3.predict(currentTree, x, labels) for x in trainset]
            testPred  = [id3.predict(currentTree, x, labels) for x in testset]
            
            trainAcc = id3.accuracy(trainPred, trainLabels)
            testAcc  = id3.accuracy(testPred, testLabels)
            
            print("Decision tree of depth", i, "using", item, "has a test accuracy of", testAcc,
                  'and a training accuracy of', trainAcc)


main()


#print(gf.best_split(dataset, labels, labels[-1], "ME"))
#tree1 = id3(dataset, labels, label_attr, labels[-1], 4, 'entropy', None)
#print("The predicted label is:", predict(tree1, dataset[1], labels))
#print("The actual label was:",dataset[1]["label"])
