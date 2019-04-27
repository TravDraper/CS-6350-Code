# CS-6350-Code
This is a Machine Learning repository created by Travis Draper for my work in CS 6350 at the University of Utah
To use the decision tree algorithms, you need to provide a csv file and corresponding dictionaries which contain all of the possible values for each parameter.  The decision trees are capable of handling numeric values by replacing the number with a marker indicating whether it is larger or smaller than the median value in the training set.

To use the ensemble learning algorithm, follow the same procedure as you would with the decision tree code.  Provide labels for the data, and choose which method you would prefer.  Note, large forests can take a long time to train.

To use the linear regression models provide a .CSV file with the output variable in the last column, The code will implement both a stochastic gradien descent, and batch gradient descent.  Both provide fairly good accuracy given that your data is somewhat linear.  


To use the perceptron models, provide a .csv file with the last column as the output variable, the outputs should be either 1,0 or 1,-1.  The different types of perceptrons can all adapt to the same type of .csv file.  You are more likely to get good accuracy out of the more complicated multi-perceptron methods.

Use of the SVM models is much the same as the perceptron, it takes in the same type of data, and outputs results in the same format.  

Logistic regression is once again the same style as the other machine learning methods in its execution and outputs.  Make sure to format your inputs in the right way.  Although the functions included in the library should work do do it on their own.

The neural networks section uses Tensorflow, more specifically the keras library to build and run a model.  I would recommend using the tutorials for those libraries included on their sites.
