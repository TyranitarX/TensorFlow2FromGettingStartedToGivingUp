# 线性回归
import numpy as np

data = []

for i in range(100):
    # -10~10的均匀分布随机数
    x = np.random.uniform(-10., 10.)
    # 正态分布随机数生成高斯噪声(正态分布的噪声)
    eps = np.random.normal(0., 0.1)
    # 实际上是目标函数
    y = 1.377 * x + 0.089 + eps

    data.append([x, y])

# 生成的数据集
data = np.array(data)


def lossFunc(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 平方差均值 loss
        totalError += (y - (w * x + b)) ** 2 / float(len(points))

        return totalError


def step_gradient(w_current, b_currnet, points, lr):
    w_gradient = 0
    b_gradient = 0

    total = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 函数 y 在 变量w上的偏导数
        w_gradient += -2 * x * (y - (w_current * x + b_currnet)) / total
        # 函数 y 在 变量b上的偏导数
        b_gradient += -2 * (y - (w_current * x + b_currnet)) / total

    # 根据当前梯度和学习率进行参数更新
    new_w = w_current - (lr * w_gradient)
    new_b = b_currnet - (lr * b_gradient)

    return new_w, new_b


def gradient_decent(w_current, b_current, points, lr, itearation):
    w = w_current
    b = b_current

    for i in range(itearation):
        w, b = step_gradient(w, b, points, lr)
        loss = lossFunc(b, w, points)
        if i % 50 == 0:
            print("iteration %d ,loss %f" % (i, loss))

    return w, b, loss


def main():
    # 学习率
    lr = 0.01
    # 初始化参数w
    w = 0
    # 初始化参数b
    b = 0
    iteartion = 1000

    w, b, loss = gradient_decent(w, b, data, lr, iteartion)

    print('final loss%f ,w%f , b%f' % (loss, w, b))


main()
