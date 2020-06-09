import argparse, os
import numpy as np
import tensorflow as tf
from math import floor
from train import train
from test import test
from read import load_train_data, load_valid_data, load_test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--gpus', type=str) #determine gpu to use
    parser.add_argument('--path', type=str) #determine path to save
    parser.add_argument('--mode', type=str) 
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
 
    if args.mode == 'train':
        data = load_train_data()
        Test_data = load_valid_data()
        #Test_data = load_test_data()
        train(data, Test_data, args.path)  
    elif args.mode == 'test':
        data = load_test_data()
        test(data, args.path)
       
            
    
