import numpy as np

# Our toy dataset

x_train = np.linspace(-3, 3, num=50)
y_train = np.cos(x_train) + np.random.normal(0, 0.1, size=50)
x_train = x_train.astype(np.float32).reshape((50, 1))
y_train = y_train.astype(np.float32).reshape((50, 1))


import tensorflow as tf
from edward.models import Normal

import matplotlib.pyplot as plt
plt.plot(x_train, y_train)

### Define a three layer bayesian NN w/ tanh activation
n=2
shape_0 = [1, n]
shape_1 = [n, n]
shape_2 = [n, 1]
#shape_3 = [2, 1]

W_0 = Normal(mu=tf.zeros(shape_0   ), sigma=tf.ones(shape_0   ))
W_1 = Normal(mu=tf.zeros(shape_1   ), sigma=tf.ones(shape_1   ))
W_2 = Normal(mu=tf.zeros(shape_2   ), sigma=tf.ones(shape_2   ))
#W_3 = Normal(mu=tf.zeros(shape_3   ), sigma=tf.ones(shape_3   ))

b_0 = Normal(mu=tf.zeros(shape_0[1]), sigma=tf.ones(shape_0[1],))
b_1 = Normal(mu=tf.zeros(shape_1[1]), sigma=tf.ones(shape_1[1],))
b_2 = Normal(mu=tf.zeros(shape_2[1]), sigma=tf.ones(shape_2[1],))
#b_3 = Normal(mu=tf.zeros(shape_3[1]), sigma=tf.ones(shape_3[1],))

x = x_train

l_0=tf.tanh(tf.matmul(  x, W_0) + b_0)
l_1=tf.tanh(tf.matmul(l_0, W_1) + b_1)
l_2=tf.matmul(l_1, W_2) + b_2

y = Normal(mu=l_2, sigma=0.1)

### Inferee
qW_0 = Normal(mu=tf.Variable(tf.zeros(shape_0   )),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_0   ))))
qW_1 = Normal(mu=tf.Variable(tf.zeros(shape_1   )),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_1   ))))
qW_2 = Normal(mu=tf.Variable(tf.zeros(shape_2   )),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_2   ))))

qb_0 = Normal(mu=tf.Variable(tf.zeros(shape_0[1])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_0[1],))))
qb_1 = Normal(mu=tf.Variable(tf.zeros(shape_1[1])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_1[1],))))
qb_2 = Normal(mu=tf.Variable(tf.zeros(shape_2[1])),
              sigma=tf.nn.softplus(tf.Variable(tf.zeros(shape_2[1],))))

### Do the variational inference with the Kullback-Leibler 

import edward as ed

data = {y: y_train}
inference = ed.KLqp({W_0: qW_0, b_0: qb_0,
                     W_1: qW_1, b_1: qb_1,
                     W_2: qW_2, b_2: qb_2}, data)
inference.run(n_iter=2000)

## Evaluate

print(ed.evaluate('mean_squared_error', data))