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
    return trn_data[0:40000], trn_labels[0:40000], tst_data, tst_labels,val_set
def pca(x3):
    x3 = x3 - x3.mean()
    u,s,vh = np.linalg.svd(x3,full_matrices = False)
    sum = s.sum()
    plt.step(np.arange(len(s)),np.cumsum(s)/sum)
    return u,s,vh

print("start")
a,b,c,d,e = load_cifar()
pca = PCA(svd_solver = 'full')
pca.fit()
filename = 'pca_model.sav'
pickle.dump(pca, open(filename, 'wb'))
v = a[7]
v_r = np.reshape(v[0:1024],(32,32))
v_g = np.reshape(v[1024:2048],(32,32))
v_b = np.reshape(v[2048:3072],(32,32))
v = cv2.merge((v_r,v_g,v_b))
plt.imshow(v,"gray")
plt.title(b[3])
plt.show()
