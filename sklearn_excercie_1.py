# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:51:52 2018

@author: Jayesh

"""
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
digits = datasets.load_digits()
digits.target
clf = svm.SVC(gamma = 0.001,C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])
clf.predict(digits.data[-1:])


#The pickle module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure. “Pickling” is the process whereby 
#a Python object hierarchy is converted into a byte stream
import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(digits.data[0:1])
digits.target[0]


























