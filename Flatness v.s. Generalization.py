# -*- coding: utf-8 -*-
"""HW3-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rO1cs1gCf_k7plKbkBe04AP5H8FSmMN6
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2

epoch = 50

"""MNIST"""

data = input_data.read_data_sets('data/MNIST/', one_hot=True);

train_num = data.train.num_examples
valid_num = data.validation.num_examples
test_num = data.test.num_examples
img_flatten = 784
img_size = 28
num_classes = 10
print("Training Dataset Size:",train_num)
print("Validation Dataset Size:",valid_num)
print("Testing Dataset Size:",test_num)

fig, axs = plt.subplots(2,5)
fig.set_size_inches(12,4)
for i in range(10):
    idx = np.where(np.argmax(data.train.labels,1)==i)[0][0]
    axs[int(i/5),i%5].imshow(data.train.images[idx].reshape(28,28))
    axs[int(i/5),i%5].set_title(str(i))
    axs[int(i/5),i%5].axis('off')

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('kernel')
    return weights

def get_bias_variable(layer_name):
    with tf.variable_scope(layer_name,reuse=True):
        bias = tf.get_variable('bias')
    return bias

"""initialize two model"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

# Model 1 
m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
m1_pool2 = tf.layers.max_pooling2d(inputs=m1_conv2,pool_size=2,strides=2);
m1_flat1 = tf.layers.flatten(m1_pool2);
m1_fc1 = tf.layers.dense(inputs=m1_flat1,units=128,activation=tf.nn.relu,name='m1_fc1');
m1_logits = tf.layers.dense(inputs=m1_fc1,units=num_classes,activation=None,name='m1_fc_out');
m1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m1_logits);
m1_loss = tf.reduce_mean(m1_cross_entropy);
m1_softmax = tf.nn.softmax(logits=m1_logits);
m1_pred_op = tf.argmax(m1_softmax,dimension=1);
m1_acc_op = tf.reduce_mean(tf.cast(tf.equal(m1_pred_op, y_cls), tf.float32));
m1_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m1_train_op = m1_optimizer.minimize(m1_loss);
#Model2
m2_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv1');
m2_pool1 = tf.layers.max_pooling2d(inputs=m2_conv1,pool_size=2,strides=2);
m2_conv2 = tf.layers.conv2d(inputs=m2_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv2');
m2_pool2 = tf.layers.max_pooling2d(inputs=m2_conv2,pool_size=2,strides=2);
m2_flat1 = tf.layers.flatten(m2_pool2);
m2_fc1 = tf.layers.dense(inputs=m2_flat1,units=128,activation=tf.nn.relu,name='m2_fc1');
m2_logits = tf.layers.dense(inputs=m2_fc1,units=num_classes,activation=None,name='m2_fc_out');
m2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m2_logits);
m2_loss = tf.reduce_mean(m2_cross_entropy);
m2_softmax = tf.nn.softmax(logits=m2_logits);
m2_pred_op = tf.argmax(m2_softmax,dimension=1);
m2_acc_op = tf.reduce_mean(tf.cast(tf.equal(m2_pred_op, y_cls), tf.float32));
m2_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m2_train_op = m2_optimizer.minimize(m2_loss);

m1_weights_conv1 = get_weights_variable('m1_conv1')
m1_weights_conv2 = get_weights_variable('m1_conv2')
m1_weights_fc1 = get_weights_variable('m1_fc1')
m1_weights_fc_out = get_weights_variable('m1_fc_out')

m1_bias_conv1 = get_bias_variable('m1_conv1')
m1_bias_conv2 = get_bias_variable('m1_conv2')
m1_bias_fc1 = get_bias_variable('m1_fc1')
m1_bias_fc_out = get_bias_variable('m1_fc_out')

m2_weights_conv1 = get_weights_variable('m2_conv1')
m2_weights_conv2 = get_weights_variable('m2_conv2')
m2_weights_fc1 = get_weights_variable('m2_fc1')
m2_weights_fc_out = get_weights_variable('m2_fc_out')

m2_bias_conv1 = get_bias_variable('m2_conv1')
m2_bias_conv2 = get_bias_variable('m2_conv2')
m2_bias_fc1 = get_bias_variable('m2_fc1')
m2_bias_fc_out = get_bias_variable('m2_fc_out')

"""run 64"""

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []

BATCH_SIZE = 64

for i in range(epoch):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc = session.run([m1_loss,m1_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, trainloss, trainacc, testloss, testacc))

m1_w_conv1,m1_w_conv2,m1_w_fc1,m1_w_fc_out = session.run([m1_weights_conv1,m1_weights_conv2,m1_weights_fc1,m1_weights_fc_out])
m1_b_conv1,m1_b_conv2,m1_b_fc1,m1_b_fc_out = session.run([m1_bias_conv1,m1_bias_conv2,m1_bias_fc1,m1_bias_fc_out])

"""run 1024"""

trainlosslist2 = []
trainacclist2 = []
testlosslist2 = []
testacclist2 = []

BATCH_SIZE = 1024
for i in range(epoch):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m2_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m2_loss,m2_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist2.append(trainloss)
    trainacclist2.append(trainacc)
    testloss, test_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist2.append(testloss)
    testacclist2.append(testacc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, trainloss, trainacc, testloss, testacc))
    
m2_w_conv1,m2_w_conv2,m2_w_fc1,m2_w_fc_out = session.run([m2_weights_conv1,m2_weights_conv2,m2_weights_fc1,m2_weights_fc_out])
m2_b_conv1,m2_b_conv2,m2_b_fc1,m2_b_fc_out = session.run([m2_bias_conv1,m2_bias_conv2,m2_bias_fc1,m2_bias_fc_out])

"""plot the loss and accuracy"""

fig,axs = plt.subplots(1,2)
fig.set_figwidth(15)
x = np.arange(epoch)
axs[0].plot(x,testacclist1,x,testacclist2)
axs[0].legend(('batch64','batch1024'))
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('accuracy')
axs[1].plot(x,testlosslist1,x,testlosslist2)
axs[1].legend(('batch64','batch1024'))
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('loss');

"""learning rate difference"""

tf.reset_default_graph() #reset

x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

### Model 1 Architecture
m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
m1_pool2 = tf.layers.max_pooling2d(inputs=m1_conv2,pool_size=2,strides=2);
m1_flat1 = tf.layers.flatten(m1_pool2);
m1_fc1 = tf.layers.dense(inputs=m1_flat1,units=128,activation=tf.nn.relu,name='m1_fc1');
m1_logits = tf.layers.dense(inputs=m1_fc1,units=num_classes,activation=None,name='m1_fc_out');
m1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m1_logits);
m1_loss = tf.reduce_mean(m1_cross_entropy);
m1_softmax = tf.nn.softmax(logits=m1_logits);
m1_pred_op = tf.argmax(m1_softmax,dimension=1);
m1_acc_op = tf.reduce_mean(tf.cast(tf.equal(m1_pred_op, y_cls), tf.float32));
m1_optimizer = tf.train.AdamOptimizer(learning_rate=0.01);
m1_train_op = m1_optimizer.minimize(m1_loss);

### Model 2 Architecture
m2_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv1');
m2_pool1 = tf.layers.max_pooling2d(inputs=m2_conv1,pool_size=2,strides=2);
m2_conv2 = tf.layers.conv2d(inputs=m2_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv2');
m2_pool2 = tf.layers.max_pooling2d(inputs=m2_conv2,pool_size=2,strides=2);
m2_flat1 = tf.layers.flatten(m2_pool2);
m2_fc1 = tf.layers.dense(inputs=m2_flat1,units=128,activation=tf.nn.relu,name='m2_fc1');
m2_logits = tf.layers.dense(inputs=m2_fc1,units=num_classes,activation=None,name='m2_fc_out');
m2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m2_logits);
m2_loss = tf.reduce_mean(m2_cross_entropy);
m2_softmax = tf.nn.softmax(logits=m2_logits);
m2_pred_op = tf.argmax(m2_softmax,dimension=1);
m2_acc_op = tf.reduce_mean(tf.cast(tf.equal(m2_pred_op, y_cls), tf.float32));
m2_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m2_train_op = m2_optimizer.minimize(m2_loss);

m1_weights_conv1 = get_weights_variable('m1_conv1')
m1_weights_conv2 = get_weights_variable('m1_conv2')
m1_weights_fc1 = get_weights_variable('m1_fc1')
m1_weights_fc_out = get_weights_variable('m1_fc_out')

m1_bias_conv1 = get_bias_variable('m1_conv1')
m1_bias_conv2 = get_bias_variable('m1_conv2')
m1_bias_fc1 = get_bias_variable('m1_fc1')
m1_bias_fc_out = get_bias_variable('m1_fc_out')

m2_weights_conv1 = get_weights_variable('m2_conv1')
m2_weights_conv2 = get_weights_variable('m2_conv2')
m2_weights_fc1 = get_weights_variable('m2_fc1')
m2_weights_fc_out = get_weights_variable('m2_fc_out')

m2_bias_conv1 = get_bias_variable('m2_conv1')
m2_bias_conv2 = get_bias_variable('m2_conv2')
m2_bias_fc1 = get_bias_variable('m2_fc1')
m2_bias_fc_out = get_bias_variable('m2_fc_out')

"""run 0.01"""

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []

BATCH_SIZE = 64

for i in range(epoch):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc = session.run([m1_loss,m1_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, trainloss, trainacc, testloss, testacc))

m1_w_conv1,m1_w_conv2,m1_w_fc1,m1_w_fc_out = session.run([m1_weights_conv1,m1_weights_conv2,m1_weights_fc1,m1_weights_fc_out])
m1_b_conv1,m1_b_conv2,m1_b_fc1,m1_b_fc_out = session.run([m1_bias_conv1,m1_bias_conv2,m1_bias_fc1,m1_bias_fc_out])

"""run 0.001"""

trainlosslist2 = []
trainacclist2 = []
testlosslist2 = []
testacclist2 = []

BATCH_SIZE = 64
for i in range(epoch):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m2_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m2_loss,m2_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist2.append(trainloss)
    trainacclist2.append(trainacc)
    testloss, test_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist2.append(testloss)
    testacclist2.append(testacc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, trainloss, trainacc, testloss, testacc))
    
m2_w_conv1,m2_w_conv2,m2_w_fc1,m2_w_fc_out = session.run([m2_weights_conv1,m2_weights_conv2,m2_weights_fc1,m2_weights_fc_out])
m2_b_conv1,m2_b_conv2,m2_b_fc1,m2_b_fc_out = session.run([m2_bias_conv1,m2_bias_conv2,m2_bias_fc1,m2_bias_fc_out])

fig,axs = plt.subplots(1,2)
fig.set_figwidth(15)
x = np.arange(epoch)
axs[0].plot(x,testacclist1,x,testacclist2)
axs[0].legend(('0.01','0.001'))
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('accuracy')
axs[1].plot(x,testlosslist1,x,testlosslist2)
axs[1].legend(('0.01','0.001'))
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('loss');