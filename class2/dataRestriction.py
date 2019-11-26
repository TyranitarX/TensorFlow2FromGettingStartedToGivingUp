import tensorflow as tf
import numpy as np

x = tf.range(9)
print(x)
# 上下边界振幅
print(tf.maximum(x, 2))
print(tf.minimum(x, 2))

conn = tf.maximum(tf.minimum(x, 7), 2)
ori = tf.clip_by_value(x, 2, 7)
print(conn)
print(ori)
