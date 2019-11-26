import tensorflow as tf
from tensorflow import keras

# 利用tensor实现一层网络
x = tf.random.normal([2, 784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x, w1) + b1
o1 = tf.nn.relu(o1)
print(o1.shape)

# 利用现有的dense
fc = keras.layers.Dense(256, activation=tf.nn.relu)
o1 = fc(x)
print(o1.shape)
