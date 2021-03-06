# -*- coding: utf-8 -*-
"""HW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13MD6P4mmQDzdpWtMJTFZfZjyYFUAHjxW
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

tf.__version__

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
y = np.sin(5*np.pi*x)/(5*np.pi*x)    
y1 = np.sign(np.sin(5*np.pi*x))  
epoch = 10000

input_x = tf.placeholder(tf.float32, [None, 1])      # input x
#input_x = tf.placeholder(tf.float32, [100, 1])     
output_y = tf.placeholder(tf.float32, [None, 1])     # input y
#output_y = tf.placeholder(tf.float32, [100, 1])

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

m0h1 = tf.layers.dense(inputs=input_x, units=5, activation=tf.nn.relu, name='m0h1')   # hidden layer
m0h2 = tf.layers.dense(inputs=m0h1, units=10, activation=tf.nn.relu, name='m0h2')        # hidden layer
m0h3 = tf.layers.dense(inputs=m0h2, units=10, activation=tf.nn.relu, name='m0h3')        # hidden layer
m0h4 = tf.layers.dense(inputs=m0h3, units=10, activation=tf.nn.relu, name='m0h4')        # hidden layer
m0h5 = tf.layers.dense(inputs=m0h4, units=10, activation=tf.nn.relu, name='m0h5') 
m0h6 = tf.layers.dense(inputs=m0h5, units=10, activation=tf.nn.relu, name='m0h6') 
m0h7 = tf.layers.dense(inputs=m0h6, units=5, activation=tf.nn.relu, name='m0h7') 
m0output = tf.layers.dense(inputs=m0h7, units=1, name='m0output')                     # output layer

loss = tf.losses.mean_squared_error(output_y, m0output)   # compute cost

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

grads = tf.gradients(loss, get_weights_variable('m0output'))[0]
grads_norm = tf.norm(grads)
print(grads)
print(grads_norm)
hessian = tf.reduce_sum(tf.hessians(loss, get_weights_variable('m0output'))[0], axis = 2)
print(hessian)
eigenv = tf.linalg.eigvalsh(hessian)
minimal_ratio = tf.divide(tf.count_nonzero(tf.greater(eigenv, 0.)),eigenv.shape[0])
print(minimal_ratio)

sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph
losslist = []
gradslist = []
minratiolist = []
for i in range(epoch):
    # train and net output
    _, l, pred, gradnorm,minratio = sess.run([train_op, loss, m0output,grads_norm,minimal_ratio], feed_dict={input_x: x, output_y: y})
    losslist.append(l)
    gradslist.append(gradnorm)
    minratiolist.append(minratio)

fig,ax = plt.subplots(1,2)
fig.set_figwidth(15)
fig.suptitle('Grad Norm and Loss')
ax[0].plot(gradslist)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('grad norm')
ax[1].plot(losslist)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('loss')
plt.pause(0.1)
plt.scatter(minratiolist,losslist)
plt.xlabel('minimal ratio')
plt.ylabel('loss');
plt.title('minimal ratio vs loss')

grad_norm_train_op = optimizer.minimize(grads_norm)

for i in range(epoch):
    # train and net output
    _, l, min_ratio = sess.run([grad_norm_train_op, loss, minimal_ratio], feed_dict={input_x: x, output_y: y})

sess = tf.Session() 
sess.run(tf.global_variables_initializer())         # initialize var in graph
gradslosslist = []
minratiolist1 = []
for i in range(1000):
    # train and net output
    _, l, pred, gradnorm,minratio = sess.run([train_op, loss, m0output,grads_norm,minimal_ratio], feed_dict={input_x: x, output_y: y})
    if i%10 == 0:
        print("Epoch: ",i,"Loss: ",l,"Minimal Ratio: ",min_ratio)
        gradslosslist.append(l)
        minratiolist1.append(min_ratio)

plt.scatter(minratiolist1,gradslosslist)
plt.xlabel('minimal ratio')
plt.ylabel('loss')