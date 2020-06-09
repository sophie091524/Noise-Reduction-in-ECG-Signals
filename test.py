import time, os, pickle
import math, random
import numpy as np
import tensorflow as tf
import numpy.matlib
from os.path import join
from graph import ImportGraph
from scipy.io import savemat
from sklearn import preprocessing


def _preprocessing(X_train):
    X_train = np.transpose(X_train)
    # [0,1]
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X_train)
    #[-1,1]
    #scaler = preprocessing.MaxAbsScaler()
    #X = scaler.fit_transform(X_train)
    X_new = np.transpose(X)
    #X_new = X
    #print(X_train.shape, X_new.shape)
    return (X_new)

def test(spec, path):
    noi = spec['noi']
    #data = _preprocessing(noi)
    data = noi
    # Clean Target
    clean = spec['clean']
    print(clean.shape)
    label = spec['label']
    #print(label)
    clean=numpy.matlib.repmat(clean,3,1)
    #data = norm
    #target = _preprocessing(clean)    
    target = clean
    print(data.shape, target.shape)
    os.system("mkdir -p "+path)

    graph = ImportGraph(path)
    err_mse = graph.err_mse(data, target)
    #for i in range(len(x)):
    #loss = graph.loss(data)
    z = graph._encode(data)
    x_h = graph._decode(z)
    data = np.reshape(data, (-1, 1024))
    x_h = np.reshape(x_h, (-1, 1024))
        #print('Loss: %.4f' % loss)
        #count += loss
    #print('MSE: %.4f COR: %.4f' % loss['pmse'], loss['corr'])
    print(err_mse)
    #print(data[0], x_h[0])
    with open('label.txt', 'w') as f:
        for lab in label:
            f.write('%s\n' %lab)
    savemat('origin', mdict={'ecg': data})
    savemat('recons', mdict={'ecg': x_h})
    savemat('clean', mdict={'ecg':target})
