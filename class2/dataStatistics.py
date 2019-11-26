import tensorflow as tf
import numpy as np

# 向量范数

x = tf.ones([2, 2])

l1 = tf.norm(x, ord=1)
l2 = tf.norm(x, ord=2)
inf = tf.norm(x, ord=np.inf)

print(l1)
print(l2)
print(inf)

#  最大最小值、均值、和
x = tf.random.normal([2, 3])

x_max = tf.reduce_max(x, axis=0)
x_min = tf.reduce_min(x, axis=0)
x_mean = tf.reduce_mean(x, axis=0)
x_sum = tf.reduce_sum(x, axis=0)

print(x)
print(x_max, x_min, x_mean, x_sum)

y = tf.random.uniform([100], dtype=tf.int64, maxval=10)
y_ = tf.random.uniform([100], dtype=tf.int64, maxval=10)
print(tf.equal(y_, y))
out = tf.cast(tf.equal(y, y_), tf.float32)
correct = tf.reduce_sum(out)
print(correct.numpy(), "%")

