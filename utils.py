import tensorflow as tf

EPSILON = 1e-5

def nn(inputs, dim, name, training):
    outputs=tf.layers.dense(inputs, dim)
    outputs = tf.layers.batch_normalization(outputs)
    outputs = tf.layers.dropout(outputs, rate=0.5)
    outputs = tf.nn.relu(outputs, name=name)
    return outputs

def cnn(inputs, channel, kernel, stride, name, training):
    #regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
    outputs = tf.layers.conv2d(inputs, channel, kernel, stride, padding='SAME')#, kernel_regularizer = regularizer)	
    outputs  = tf.layers.max_pooling2d(inputs, poolsize, strides, padding='SAME')
    outputs = tf.layers.batch_normalization(outputs)
    #outputs = tf.layers.dropout(outputs, rate=0.5)
    outputs = tf.nn.elu(outputs, name=name)
    return outputs

def cnn_trans(inputs, channel, kernel, stride, name, training):
    #regularizer = tf.contrib.layers.l2_regularizer(scale=5e-4)
    outputs = tf.layers.conv2d_transpose(inputs, channel, kernel, stride, padding='SAME')#, kernel_regularizer = regularizer)
    outputs = tf.layers.batch_normalization(outputs)
    #outputs = tf.layers.dropout(outputs, rate=0.5)
    outputs = tf.nn.elu(outputs, name=name)
    return outputs

def l1(x, y, name='l1_loss'):
    with tf.name_scope(name):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(x-y), axis=1))

def mse(x, y, name='mse'):
    with tf.name_scope(name):
        mse = tf.reduce_mean(tf.squared_difference(x, y))
    return mse

def p_mse(x, y, name='PMSE'):
    with tf.name_scope(name):
        mse = tf.reduce_sum(tf.square(x - y), axis=1)
        std_x = tf.reduce_sum(tf.square(x), axis=1)
        PRD = tf.reduce_mean(tf.sqrt(mse/std_x))
    return PRD

def cc(x, y, name='Correlation'):
    with tf.name_scope(name):
        mean_x = tf.reduce_mean(x, axis=1, keep_dims=True)
        mean_y = tf.reduce_mean(y, axis=1, keep_dims=True)
        cov_xy = tf.reduce_sum((x-mean_x)*(y-mean_y), axis=1)
        std_x = tf.reduce_sum(tf.square(x - mean_x), axis=1)
        std_y = tf.reduce_sum(tf.square(y - mean_y), axis=1)
        cor = tf.reduce_mean(cov_xy/tf.sqrt(std_x*std_y))
    return cor

        

def GaussianSample(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))

def logpx(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = tf.log(2. * PI)
        var = tf.exp(log_var)
        x_mu2 = tf.square(x - mu)   # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob

def kld(mu1, lv1, mu2, lv2):
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)

def _compute_gradient_penalty(self, J, x, scope_name='GradientPenalty'):
    with tf.name_scope(scope_name):
        grad = tf.gradients(J, x)[0]  # as the output is a list, [0] is needed
        grad_square = tf.square(grad)
        grad_squared_norm = tf.reduce_sum(grad_square, axis=1)
        grad_norm = tf.sqrt(grad_squared_norm)
        # penalty = tf.square(tf.nn.relu(grad_norm - 1.)) # FIXME: experimental
        penalty = tf.square(grad_norm - 1.)
    return tf.reduce_mean(penalty)


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob

