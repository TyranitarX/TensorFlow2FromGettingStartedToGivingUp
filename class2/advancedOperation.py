import tensorflow as tf
import numpy as np

# 第一维为班级 第二维为学生 第三维为8个科目
x = tf.random.uniform([4, 35, 8], maxval=100, dtype=tf.int64)

print(x.shape)

xs = tf.gather(x, [0, 2], axis=0)

print(xs.shape)

# tf.gather_nd 可以通过多维坐标收集数据
# 获取第一个班第一个学生 第二个班第二个学生 第三个班第三个学生
gather = tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]])
print(gather.shape)

# tf.boolean_mask 根据指定掩码采样
mask = [True, False, False, True]

mask_get = tf.boolean_mask(x, mask, axis=0)

print(mask_get.shape)

# tf.where 根据cond条件从A和B中读取数据。cond 为True从A中读取数据,cond为False时从B中读取数据。
x = tf.ones([3, 3])
y = tf.zeros([3, 3])
cond = tf.constant([[True, True, False], [False, True, True], [True, False, True]])
# 若无x，y参数 则返回cond为True时的索引
z = tf.where(cond)
z = tf.where(cond, x, y)

print(z)
