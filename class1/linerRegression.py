import tensorflow as tf
import numpy as np

data = []

for i in range(100):
    x = np.random.uniform(-10., 10.)

    eps = np.random.normal(0., 0.1)

    y = 1.377 * x + 0.089 + eps

    data.append([x, y])

data = np.array(data)


def lossFunc(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2 / float(len(points))

        return totalError


def step_gradient(w_current, b_currnet, points, lr):
    w_gradient = 0
    b_gradient = 0

    total = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += -2 * x * (y - (w_current * x + b_currnet)) / total
        b_gradient += -2 * (y - (w_current * x + b_currnet)) / total

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
    lr = 0.01
    w = 0
    b = 0
    iteartion = 1000

    w, b, loss = gradient_decent(w, b, data, lr, iteartion)

    print('final loss%f ,w%f , b%f' % (loss, w, b))


main()
