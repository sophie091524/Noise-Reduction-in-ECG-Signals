from __future__ import division
from utils import (nn, cnn, cnn_trans, GaussianSample,
                   _compute_gradient_penalty, GaussianLogDensity,
                   kld, logpx, p_mse, cc, l1, mse)
import tensorflow as tf

z_dim = 32
input_dim = 1024

class DAE:
    def __init__(self):
        self._encode = tf.make_template('Enc', self._encoder)
        self._decode = tf.make_template('Dec', self._decoder)

    def _encoder(self, x, training=False):
        with tf.name_scope('G'):
            x = tf.reshape(x, shape=(-1, input_dim, 1, 1))
            ######## FCN
            conv1 = cnn(x, 40, [16, 1], [2, 1], 'conv1', training)
            conv2 = cnn(conv1, 20, [16, 1], [2, 1], 'conv2', training)
            conv3 = cnn(conv2, 20, [16, 1], [2, 1], 'conv3', training)
            conv4 = cnn(conv3, 20, [16, 1], [2, 1], 'conv4', training)
            conv5 = cnn(conv4, 40, [16, 1], [2, 1], 'conv5', training)
            conv6 = cnn(conv5, 1, [16, 1], [1, 1], 'conv6', training)
            outputs = conv6
            #print(conv1.shape,conv2.shape,conv3.shape,conv4.shape,conv5.shape,conv6.shape)
            outputs = tf.reshape(outputs, shape=(-1, z_dim))
            #print(outputs.get_shape())
            ######## DNN
            #h1 = nn(x, 512, 'h1', training)
            #h2 = nn(h1, 256, 'h2', training)
            #h3 = nn(h2, 128, 'h3', training)
            #h4 = nn(h3, 64, 'h4', training)
            #h5 = nn(h4, z_dim, 'h5', training)
            #h6 = nn(h5, z_dim, 'h6', training)
            #outputs = h6
            #print(outputs.get_shape())
        return outputs

    def _decoder(self, x, training=False):
        with tf.name_scope('D'):
            x = tf.reshape(x, shape=(-1, z_dim, 1, 1))
            ######## FCN
            conv1 = cnn_trans(x, 1, [16, 1], [1, 1], 'conv1', training)
            conv2 = cnn_trans(conv1, 40, [16, 1], [2, 1], 'conv2', training)
            conv3 = cnn_trans(conv2, 20, [16, 1], [2, 1], 'conv3', training)
            conv4 = cnn_trans(conv3, 20, [16, 1], [2, 1], 'conv4', training)
            conv5 = cnn_trans(conv4, 20, [16, 1], [2, 1], 'conv5', training)
            conv6 = cnn_trans(conv5, 40, [16, 1], [2, 1], 'conv6', training)
            conv7 = tf.layers.conv2d(conv6, 1, [16,1], [1, 1], padding='SAME')
            #outputs = tf.nn.tanh(conv6, 'outputs')
            outputs = conv7
			#print(conv1.shape,conv2.shape,conv3.shape,conv4.shape,conv5.shape,conv6.shape,conv7.shape)
            ########## CNN ##########
            #h1 = tf.contrib.layers.flatten(conv5)
            #print(h1.shape)
            #h2 = nn(h1, 1024, 'h2', training)
            #outputs = tf.layers.dense(h2, input_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #outputs = tf.layers.dense(h1, input_dim)
            #outputs = tf.layers.batch_normalization(outputs)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #h1 = tf.contrib.layers.flatten(conv4)
            #print(h1.shape)
            #h2 = nn(h1, 512,'h2', training)
            #outputs = tf.layers.dense(h2, input_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #outputs = tf.nn.tanh(outputs, 'output')
            #outputs = tf.layers.dense(outputs,1024)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #outputs = tf.layers.dense(outputs, input_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #outputs = conv7
            #print(outputs.shape)
            #outputs = tf.reshape(outputs, shape=(-1, input_dim))
            #print(outputs.get_shape())
            ######## DNN
            #h1 = nn(x, z_dim, 'h1', training)
            #h2 = nn(h1, 64, 'h2', training)
            #h3 = nn(h2, 128, 'h3', training)
            #h4 = nn(h3, 256, 'h4', training)
            #h5 = nn(h4, 512, 'h5', training)
            #h6 = nn(h5, input_dim, 'h6', training)
            #outputs = tf.layers.dense(h6, input_dim)
            #outputs = tf.layers.dropout(outputs, rate=0.5)
            #h7 = nn(h6, input_dim, 'h7', training)
            #outputs = h7
            #print(outputs.get_shape())
        return outputs

    def loss(self, x, y, training=True):
        with tf.name_scope('loss'):
            z = self._encode(x, training=training)
            x_h = self._decode(z, training=training)
            loss = dict()
            #loss['pmse'] = p_mse(x, x_h)
            #loss['corr'] = cc(x, x_h)
            #loss['diff'] = l1(x, x_h)
            loss['mse'] = mse(y, x_h)
            #tf.summary.scalar('pmse', loss['pmse'])
            #tf.summary.scalar('corr', loss['corr'])
            #tf.summary.scalar('diff', loss['diff'])
            tf.summary.scalar('mse', loss['mse'])
        return loss

    def encode(self, x):
        return self._encode(x)

    def decode(self, x):
        return self._decode(x)


