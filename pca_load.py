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
    val_set = trn_data[40000:50000]
    return trn_data[0:40000], trn_labels[0:40000], tst_data, tst_labels,val_set,trn_labels[40000:50000]

trn_data,trn_labels,tst_data,tst_labels,val_set,val_labels = load_cifar()
filename = 'pca_model.sav'
pca_model = pickle.load(open(filename, 'rb'))
# print(len(pca_model.singular_values_))
## 1500 looks to be a good number
A = pca_model.singular_values_
V = pca_model.components_
print(V[:,0:1500].shape)
mul_ = np.matmul(trn_data,V[:,0:1500])
filename = 'reduced_features.sav'
# pca_model = pickle.dump(open(filename, 'wb'))
pickle.dump(mul_, open(filename, 'wb'))
print(mul_.shape)
# sum = A.sum()
# plt.step(np.arange(len(A)),np.cumsum(A)/sum)
# plt.show()
print(A)