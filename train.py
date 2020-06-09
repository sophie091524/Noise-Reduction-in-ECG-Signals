import time, os, pickle
import math, random
import numpy as np
#import numpy random.randint
import numpy.matlib
import tensorflow as tf
from model import(z_dim, input_dim, DAE)
from sklearn import preprocessing

epoch = 10000
batch_num = 32

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

def train(data, Test_data, path):
    os.system("mkdir -p " + path)
    #print(path)
    noi = data['noi']
    #feat = _preprocessing(noi)
    feat = noi
    # Train Clean Target
    clean = data['clean']
    #print(clean.shape)
    #print(int(clean.shape[0]))
    clean = numpy.matlib.repmat(clean,5,1)
    #print(feat.shape, clean.shape)
    #target = _preprocessing(clean)
    target = clean
    #print(feat.shape)
    # validation data
    Test_noi = Test_data['noi']
    #Test_feat = _preprocessing(Test_noi)
    Test_feat = Test_noi
    # valid Clean Target
    Test_clean = Test_data['clean']
    clean_Test=numpy.matlib.repmat(Test_clean,5,1)
    #Test_target= _preprocessing(clean_Test)
    Test_target = clean_Test
    #print(Test_feat.shape, Test_target.shape)

    x = tf.placeholder(tf.float32, [None, input_dim], name='x')
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')
    y = tf.placeholder(tf.float32, [None, input_dim], name='y')

    model = DAE()
    loss = model.loss(x,y)
    training_summary = tf.summary.scalar("training_loss", loss['mse'])
    validation_summary = tf.summary.scalar("validation_loss", loss['mse'])
    
    _z = model.encode(x)
    x_h = model.decode(z)
    tf.add_to_collection('encode', _z)
    tf.add_to_collection('decode', x_h)
    tf.add_to_collection('err_mse', loss['mse'])
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #reg_constant = 0.01
    #l2_loss = tf.losses.get_regularization_loss()
    #w_loss = loss['mse'] + l2_loss # + reg_constant * sum(reg_losses)
    w_loss = loss['mse']

    #lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables if 'bias' not in v.name ]) * 0.001
    #w_loss = loss['mse'] + lossL2
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    optimize = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.9, beta2=0.99).minimize(w_loss)
    #optimize = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(loss)
    #merged = tf.summary.merge_all()
     

    with tf.Session() as sess:
        start = time.time()
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(path+'logs', sess.graph)
        patience = 10
        best_loss = 10000.
        for i in range(epoch):
            Num= np.arange(feat.shape[0])
            np.random.shuffle(Num)
            
            feat=feat[Num,:]
            target=target[Num,:]
            
            #length = int(feat.shape[0]/2)
            feat_1 = feat[0:-int(feat.shape[0]/2),:]
            feat_2 = feat[-int(feat.shape[0]/2):,:]
            com = np.array([feat_1,feat_2])
            target_1 = target[0:-int(target.shape[0]/2),:]
            target_2 = target[-int(target.shape[0]/2):,:]
            com_target = np.array([target_1,target_2])
            #print(feat_1.shape, feat_2.shape, com.shape)
            #feat_1 = np.reshape(feat[0:-int(feat.shape[0]/2),:], (int(feat.shape[0]/2),1024,-1))
            
            #feat_2 = np.reshape(feat[-int(feat.shape[0]/2):,:], (int(feat.shape[0]/2),1024,-1))
          
            for k in range(2):
                feat_noi = np.reshape(com[k,:,:], (-1,1024))
                #print(feat.shape, com[k,:,:].shape)
                #feat_noi = com[k,:,:]
                target_clean = com_target[k,:,:]
                #print(feat_noi.shape)
                for j in range(0, len(feat_noi), batch_num):
                    sess.run([optimize], feed_dict={x:feat_noi[j:j+batch_num], y:target_clean[j:j+batch_num]})
                if i%10 == 0:
                    # To log training accuracy.   
                    err, train_summ = sess.run([loss, training_summary],
                                          feed_dict={x:feat_noi,y:target_clean})
                    writer.add_summary(train_summ, i) 
                                                        
                    # To log validation accuracy.
                    Test_err, valid_summ= sess.run([loss,validation_summary],
                                          feed_dict={x:Test_feat, y:Test_target})                                                                                                                                
                    writer.add_summary(valid_summ, i)  
                    if Test_err['mse'] < best_loss:
                        best_loss = Test_err['mse']
                        patience = 10
                        saver.save(sess, path+'test_best_model')
                    else:
                        patience -= 1            
                    print('Epoch [%4d] Time [%5.4f] MSE [%.6f] Val_MSE [%.6f]'%(i+1, time.time() - start, err['mse'], Test_err['mse']))
            if patience == 0:
                print('Early Stopping')
                break

