import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def print_activations(t):
    print(t.op.name,'',t.get_shape().as_list())

def inference(images,noise,image_size,batch_size,num_pattrens,scale):
    parameters = []

    with tf.name_scope('encode') as scope:
        noise_image = images + scale*noise
        kernel = tf.Variable(tf.compat.v1.random.truncated_normal([image_size, image_size, 1, num_pattrens],
                                                 dtype=tf.float32, stddev=1e-1), name='weights1')
        conv = tf.nn.conv2d(noise_image, kernel, [1, 1, 1, 1], padding='VALID', name=scope)
        conv1 = conv
        print_activations(conv1)
        parameters += [kernel]

    with tf.name_scope('batchNormal') as scope:
        bacthNormal = tf.layers.batch_normalization(conv1, momentum=0.9, name=scope)

    with tf.name_scope('deconv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([9,9,num_pattrens,num_pattrens],
                    dtype=tf.float32, stddev=1e-1) ,name='weights')
        deconv1 = tf.nn.conv2d_transpose(bacthNormal, kernel,
                    [batch_size,9,9,num_pattrens], [1,2,2,1], padding="VALID", name = scope)
        parameters += [kernel]
        print_activations(deconv1)

    with tf.name_scope('fullyconnect1') as scope:
        reshape = tf.layers.flatten(deconv1)
        dim = reshape.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal([dim,400],
                    dtype=tf.float32, stddev=1e-1) ,name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                            trainable=True, name='biases')
        local = tf.nn.relu(tf.matmul(reshape,weight)+biases, name='local')
        parameters += [weight, biases]
        print_activations(local)

    with tf.name_scope('fullyconnect2') as scope:
        dim = local.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal([dim,200],
                    dtype=tf.float32, stddev=1e-1) ,name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                            trainable=True, name='biases')
        local2 = tf.nn.relu(tf.matmul(local,weight)+biases, name='local2')
        parameters += [weight, biases]
        print_activations(local2)

    with tf.name_scope('fullyconnect3') as scope:
        dim = local2.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal([dim,100],
                    dtype=tf.float32, stddev=1e-1) ,name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                            trainable=True, name='biases')
        local3 = tf.nn.relu(tf.matmul(local2,weight)+biases, name='loca3')
        parameters += [weight, biases]
        print_activations(local3)

    with tf.name_scope('fullyconnect4') as scope:
        dim = local3.get_shape()[1].value
        weight = tf.Variable(tf.truncated_normal([dim,10],
                    dtype=tf.float32, stddev=1e-1) ,name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32),
                            trainable=True, name='biases')
        local4 = tf.nn.relu(tf.matmul(local3,weight)+biases, name='loca4')
        parameters += [weight, biases]
        print_activations(local4)

    return local4, parameters


