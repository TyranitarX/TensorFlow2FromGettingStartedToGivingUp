import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets

(x, y), (x_test, y_test) = datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices((x, y))

dataset_shuffle = dataset.shuffle(10000)


def process(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)

    return x, y

dataset_shuffle = dataset_shuffle.repeat(20)
dataset_shuffle = dataset_shuffle.batch(128)
dataset_shuffle = dataset_shuffle.map(process)

print(dataset_shuffle)
# for (x,y) in (dataset_shuffle):
# for step,(x,y) in enumerate(dataset_shuffle):
