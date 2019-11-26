import tensorflow as tf
import numpy as np
from tensorflow import keras

# 参数方式实现正向传播
x = tf.random.normal([2, 784])

w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))

w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))

w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))

w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

with tf.GradientTape() as tape:
    h1 = x @ w1 + b1
    h1 = tf.nn.relu(h1)
    h2 = h1 @ w2 + b2
    h2 = tf.nn.relu(h2)
    h3 = h2 @ w3 + b3
    h4 = h3 @ w4 + b4

    print(h4.shape)

# dense方式实现正向传播
fc1 = keras.layers.Dense(256, activation=tf.nn.relu)
fc2 = keras.layers.Dense(128, activation=tf.nn.relu)
fc3 = keras.layers.Dense(64, activation=tf.nn.relu)
fcout = keras.layers.Dense(10)

h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
out = fcout(h3)

print(out.shape)


