import os
import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random
import time
import importlib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *

from functools import wraps
from time import time as _timenow
from sys import stderr
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
def load_cifar():

    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    for i in trange(0,5):
        batchName = './data/data_batch_{0}'.format(i + 1)
        print(batchName)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    tst_data.extend(unpickled['data'])
    tst_labels.extend(unpickled['labels'])
    print(len(trn_data[0:40000]))
    print(len(tst_data))
    val_data = trn_data[40000:50000]
    return trn_data[0:40000], trn_labels[0:40000], tst_data, tst_labels,val_data,trn_labels[40000:50000]

def logRegression(trn_data,trn_labels,val_set):
    logreg = LogisticRegression(penalty = 'l2')
    logreg.fit(trn_data,trn_labels)
    val_pred = logreg.predict(val_set)
    return val_pred

trn_data,trn_labels,tst_data,tst_labels,val_data,val_labels = load_cifar()
filename = 'reduced_features.sav'
feature_vector = pickle.load(open(filename, 'rb'))
pca = PCA(svd_solver = 'full')
pca.fit(val_data)
filename = 'val_data_pca.sav'
pickle.dump(pca, open(filename, 'wb'))
y_pred = logRegression(feature_vector,trn_labels,)
cnf_matrix = metrics.confusion_matrix(val_labels, y_pred)
