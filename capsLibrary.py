# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:23:39 2018

@author: Ayush Bansal
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# Also Fell like code is a bit slow and need to be improved

data = input_data.read_data_sets('./data/mnist', one_hot=True)

# Configurations
BATCH_SIZE = 64
EPOCHS = 1
NUMBER_OF_ROUTING = 2
LEARNING_RATE = 1e-3

# Parameters used similar to in research paper
lam = 0.5
mplus = 0.9
mmin = 0.1

def squash(capsule):
    norm = tf.norm(capsule, axis=2)
    factor = tf.expand_dims(tf.divide(norm, tf.add(1.0, tf.square(norm))), 2)
    return tf.multiply(capsule, factor)


def routing(uhat_vector, b_values):
    for _ in range(NUMBER_OF_ROUTING):
        c_values = tf.nn.softmax(b_values, 1)
        s_vector = tf.reduce_sum(tf.multiply(c_values, uhat_vector), axis=1)
        v_vector = squash(capsule=s_vector)
        temp = tf.expand_dims(v_vector, 1)
        updated_values = tf.reduce_mean(tf.reduce_sum(tf.multiply(uhat_vector, temp), axis=3), axis=0)
        updated_values = tf.expand_dims(tf.expand_dims(updated_values, 0), -1)
        b_values = tf.add(b_values, updated_values)
    return v_vector


def primaryCaps(conv_tensor, kernel_size, no_of_kernels, stride=[1, 1], padding=0, capsule_length=8):
    conv = tf.layers.conv2d(conv_tensor, no_of_kernels, kernel_size, stride)
    no_of_channels = int(no_of_kernels / capsule_length)
    feature_map_height, feature_map_width = conv.shape[1].value, conv.shape[2].value
    capsule = tf.reshape(conv, shape=[-1, feature_map_height*feature_map_width*no_of_channels, capsule_length])
    squashedCapsules = squash(capsule=capsule)
    return squashedCapsules


def fully_connected_layer(u_vector, output_num_capsules, output_capsule_length):
    number_input_capsules = u_vector.shape[1].value
    input_capsule_length = u_vector.shape[2].value
    u_vector = tf.expand_dims(u_vector, 2)
    conversionMatrix = tf.Variable(tf.random_normal([number_input_capsules, input_capsule_length, output_num_capsules*output_capsule_length]))
    conversionMatrix = tf.tile(tf.expand_dims(conversionMatrix, 0), [tf.shape(u_vector)[0], 1, 1, 1])
    uhat_vector = tf.reshape(tf.matmul(u_vector, conversionMatrix), shape=[-1, number_input_capsules, output_num_capsules, output_capsule_length])
    b_values = tf.Variable(tf.zeros(shape=[1, number_input_capsules, output_num_capsules, 1]))
    v_vector = routing(uhat_vector, b_values)
    return v_vector


# This would be used to calculate length of the vector
def get_prob(v_vector):
    return tf.norm(v_vector, axis=2)


# For now using same reconstruction layer as that of Hinton's Paper
def reconstruction(capsule):
    fc1 = tf.layers.dense(capsule, 512, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc2, 784, activation=tf.nn.relu)
    return fc3


X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x = tf.reshape(X, shape=[-1, 28, 28, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

conv_tensor = tf.layers.conv2d(x, filters=256, kernel_size=[9, 9])
u_vector = primaryCaps(conv_tensor, kernel_size=[9, 9], no_of_kernels=256, capsule_length=8, stride=[2, 2])
v_vector = fully_connected_layer(u_vector, 10, 16)
probabilities = get_prob(v_vector)

# Still not added the reconstruction Loss

loss = tf.reduce_sum(tf.add(tf.multiply(y, tf.square(tf.maximum(0.0, mplus - probabilities))), lam * tf.multiply((1 - y), tf.square(tf.maximum(0.0, probabilities - mmin)))))
train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

result = tf.argmax(probabilities, axis=1)
actual = tf.argmax(y, axis=1)
correct_prediction = tf.equal(result, actual)
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()


with tf.Session() as sess:
    isTraining = False
    if isTraining:
        if os.path.exists('./data/model.ckpt'):
            saver.restore(sess, "./data/model.ckpt")
            print('Model Restored')
        else:
            sess.run(tf.global_variables_initializer())
        num_batch = int(data.train.num_examples / BATCH_SIZE)
        for step in range(EPOCHS):
            for batch in range(num_batch):
                batch_x, batch_y = data.train.next_batch(BATCH_SIZE)
                sess.run(train, feed_dict={X: batch_x, y: batch_y})
                print('Completed Batch: {}'.format(batch))
            print('iteration {} completed'.format(step))
            save_path = saver.save(sess, "./data/model{}.ckpt".format(step))
        print('Training Completed')
        print('Model Saved')
    else:
        saver.restore(sess, "./data/model0.ckpt")
        print('Model Restored')

    num_batch = int(data.test.num_examples / BATCH_SIZE)
    total = 0
    for batch in range(num_batch):
        test_x, test_y = data.test.next_batch(BATCH_SIZE)
        acc = sess.run(accuracy, feed_dict={X: test_x, y: test_y})
        total += acc
        print('Batch {} completed with accuracy: {}'.format(batch, acc/BATCH_SIZE))
    print('Accuracy: {}'.format(total/data.test.num_examples))
