#!/usr/bin/env python
#
# file: unm_rf_open.py
#
# descr: reads two csv files and performs an evaluation.
#
# usage: unm_rf_open train.csv test.csv
#
# import system libraries
#
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# define important constants
#
ISIP_MAGIC = int(27)    # random number seed
CHAN_LABEL = 'O1'   # the channel label
TEST_SIZE = float(0.2)  # the size of the test set as a percentage

# import the data
#
df_train = pd.read_csv(sys.argv[1], header = None)
df_test = pd.read_csv(sys.argv[2], header = None)

# extract the features:
#  the first column is the filename/label
#  the second column is the channel
#
X_train = df_train.iloc[:, 2:102].values
X_test  = df_test.iloc[:, 2:102].values

#---
# train: convert the filenames to labels
#
labels_train = []
Xtmp_train = []
for i in range(len(df_train)):

    # check for the channel label
    #
    if CHAN_LABEL in df_train.iloc[i,1]:

        Xtmp_train.append(X_train[i])

        # convert the label to an integer value
        #
        parts = df_train.iloc[i,0].split("/")
        label = parts[-2]
        if label=='normal':
            labels_train.append(0)
        elif label=='abnormal':
            labels_train.append(1)
        else:
            print("*> error: unknown label")
            print(label)
            print(parts)
            exit(1)

#---
# test: convert the filenames to labels
#
labels_test = []
Xtmp_test = []
for i in range(len(df_test)):

    # check for the channel label
    #
    if CHAN_LABEL in df_test.iloc[i,1]:

        Xtmp_test.append(X_test[i])

        # convert the label to an integer value
        #
        parts = df_test.iloc[i,0].split("/")
        label = parts[-2]
        if label=='normal':
            labels_test.append(0)
        elif label=='abnormal':
            labels_test.append(1)
        else:
            print("*> error: unknown label")
            print(label)
            print(parts)
            exit(1)

# convert the list to a numpy array
#
Xchan_train = np.array(Xtmp_train)
Xchan_test = np.array(Xtmp_test)

# convert the labels into a vector for SkLearn
#        
y_train = np.array(labels_train)
y_test = np.array(labels_test)
    
# initialize the KNN classifier
#
knn_classifier = KNeighborsClassifier(n_neighbors=1)

#--------------------
# Closed Set Testing: train and evaluate on all the data
#--------------------

# train the classifier
#
knn_classifier.fit(Xchan_train, y_train)

# predict the labels for both sets
#
y_pred_train = knn_classifier.predict(Xchan_train)
y_pred_test = knn_classifier.predict(Xchan_test)

# calculate the accuracy of the classifier on both sets
#
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# calculate the confusion matrix
#
confusion_mat_train = confusion_matrix(y_train, y_pred_train)
confusion_mat_test = confusion_matrix(y_test, y_pred_test)

# display the results
#
print("")
print("Accuracy (Train): %10.4f%%" % (accuracy_train * 100.0))
print("Confusion Matrix (Train):")
print(confusion_mat_train)
print("Accuracy (Test): %10.4f%%" % (accuracy_test * 100.0))
print("Confusion Matrix (Test):")
print(confusion_mat_test)
print("")

#
# end of file
