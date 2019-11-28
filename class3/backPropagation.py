import tensorflow as tf
from tensorflow.keras import layers
x = tf.constant(1.)
w1 = tf.constant(2.)
b1 = tf.constant(1.)
w2 = tf.constant(2.)
b2 = tf.constant(1.)

with tf.GradientTape(persistent=True) as tape:
    tape.watch([w1, w2, b1, b2])
    y1 = x * w1 + b1
    y2 = y1 * w2 + b2

dy2_dy1 = tape.gradient(y2, y1)
dy1_dw1 = tape.gradient(y1, w1)
dy2_dw1 = tape.gradient(y2, w1)
print(dy2_dy1 * dy1_dw1)
print(dy2_dw1)

layers.Dense()




