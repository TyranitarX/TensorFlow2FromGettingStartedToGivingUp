import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.constant([2, 3, 4, 5])
print(b)
b = tf.pad(b, [[2, 0]])
print(b)

c = tf.stack([a, b], axis=0)

print(c)

b = tf.tile(b, [2])
print(b)
