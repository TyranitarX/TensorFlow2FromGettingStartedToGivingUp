import tensorflow as tf
import numpy as np

# 标量
i = tf.constant(1.1)
j = tf.constant([1.1, 2, 3])
matrix = tf.constant([[1.1, 1.2], [2.1, 2.2]])
string = tf.constant('123')

# 变量
x1 = tf.constant(1.1)

x2 = tf.Variable(x1)

# 合并

x11 = tf.random.normal([1, 2, 3])
x22 = tf.random.normal([2, 2, 3])

x33 = tf.concat([x11, x22], axis=0)
print(x33.shape)
a = tf.random.normal([10, 10, 3])
b = tf.random.normal([10, 10, 3])
a1 = tf.random.normal([10, 10, 3])
a2 = tf.random.normal([10, 10, 3])
a3 = tf.random.normal([10, 10, 3])
a4 = tf.random.normal([10, 10, 3])
c = tf.stack([a, b, a1, a2, a3, a4], axis=0)

print(c.shape)
# 分割
d = tf.split(c, axis=0, num_or_size_splits=[1, 2, 3])

print(len(d))
print(d[0].shape)

e = tf.unstack(c, axis=0)

print(len(e))
print(e[0].shape)


