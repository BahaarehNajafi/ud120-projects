#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
 
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
clf = SVC(C=10000, kernel='rbf')

t1 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t1, 3), "s")

t2 = time()
pred = clf.predict(features_test)
print("test time:", round(time()-t2, 3), "s")

acc = accuracy_score(labels_test, pred)
print(acc)

print(pred[10])
print(pred[26])
print(pred[50])
