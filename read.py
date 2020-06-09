from __future__ import print_function
import numpy as np
import pickle

def load_train_data():
    NODES = ['0db', '2.5db', '5db', '7.5db']
    noi = np.load('./dataset/-2.5db/train.npy')
    clean = np.load('./dataset/clean/train.npy')
    for n in NODES:
        data_noi = np.load('./dataset/' + n + '/train.npy')
        noi = np.concatenate([noi, data_noi], axis=0)
    return dict(noi=noi, clean=clean)            

def load_valid_data():
    NODES = ['0db', '2.5db', '5db', '7.5db']
    noi = np.load('./dataset/-2.5db/val.npy')
    clean = np.load('./dataset/clean/val.npy')
    for n in NODES:
        data_noi = np.load('./dataset/' + n + '/val.npy')
        noi = np.concatenate([noi, data_noi], axis=0)
    return dict(noi=noi, clean=clean)

def load_test_data():
    NODES =  ['3db', '7db']
    noi = np.load('./dataset/-1db/test.npy')
    noi_lab = np.load('./dataset/-1db/test_name.npy')
    clean = np.load('./dataset/clean/test.npy')
    for n in NODES:
        data_noi = np.load('./dataset/' + n + '/test.npy')
        noi = np.concatenate([noi, data_noi], axis=0)
        #data_noi_lab = np.load('./dataset/'+ n +'/test_name.npy')
        #data_noi_lab = np.reshape(data_noi_lab, (-1,1))
        #label = np.concatenate([noi_lab, data_noi_lab], axis = 0)
    label = noi_lab
    label = np.reshape(label, (-1,1))
    #print(label,noi.shape, clean.shape)
    return dict(noi=noi, clean=clean, label=label)
