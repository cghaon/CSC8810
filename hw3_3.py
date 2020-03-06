# -*- coding: utf-8 -*-
"""HW3-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1re9O1TPn_wGhuYvBEJPR2g62Dn-dbeFQ
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2

BATCH_SIZE = [4,16,64,256,512,1024,2048]

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

"""M1"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=36,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
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
sens_op = tf.norm(tf.gradients(m1_loss,input_x))

"""run different batch"""



session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []
sensitivity1 = []


for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc, sen = session.run([m1_loss,m1_acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    sensitivity1.append(sen)
    msg = "batch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}, Sensitivisy:{5:>6.1%} "
    print(msg.format(i, trainloss, trainacc, testloss, testacc, sen))

fig,axs = plt.subplots(1,1)
fig.set_figwidth(15)
x = BATCH_SIZE
print(x)
print(testlosslist1)
axs.plot(x,testlosslist1,x,testacclist1,x,sensitivity1)
axs.legend(('loss','accuracy','sensitivity'))
axs.set_xlabel('batch')
axs.set_ylabel('variables')

"""model2"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=4,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=2,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
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
sens_op = tf.norm(tf.gradients(m1_loss,input_x))

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []
sensitivity1 = []


for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc, sen = session.run([m1_loss,m1_acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    sensitivity1.append(sen)
    msg = "batch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}, Sensitivisy:{5:>6.1%} "
    print(msg.format(i, trainloss, trainacc, testloss, testacc, sen))
    
fig,axs = plt.subplots(1,1)
fig.set_figwidth(15)
x = BATCH_SIZE
print(x)
print(testlosslist1)
axs.plot(x,testlosslist1,x,testacclist1,x,sensitivity1)
axs.legend(('loss','accuracy','sensitivity'))
axs.set_xlabel('batch')
axs.set_ylabel('variables')

"""3"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=4,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=2,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
m1_pool2 = tf.layers.max_pooling2d(inputs=m1_conv2,pool_size=2,strides=2);
m1_conv3 = tf.layers.conv2d(inputs=m1_pool2,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv3');
m1_pool3 = tf.layers.max_pooling2d(inputs=m1_conv3,pool_size=2,strides=2);
m1_flat1 = tf.layers.flatten(m1_pool3);
m1_fc1 = tf.layers.dense(inputs=m1_flat1,units=128,activation=tf.nn.relu,name='m1_fc1');
m1_logits = tf.layers.dense(inputs=m1_fc1,units=num_classes,activation=None,name='m1_fc_out');
m1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m1_logits);
m1_loss = tf.reduce_mean(m1_cross_entropy);
m1_softmax = tf.nn.softmax(logits=m1_logits);
m1_pred_op = tf.argmax(m1_softmax,dimension=1);
m1_acc_op = tf.reduce_mean(tf.cast(tf.equal(m1_pred_op, y_cls), tf.float32));
m1_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m1_train_op = m1_optimizer.minimize(m1_loss);
sens_op = tf.norm(tf.gradients(m1_loss,input_x))

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []
sensitivity1 = []


for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc, sen = session.run([m1_loss,m1_acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    sensitivity1.append(sen)
    msg = "batch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}, Sensitivisy:{5:>6.1%} "
    print(msg.format(i, trainloss, trainacc, testloss, testacc, sen))
    
fig,axs = plt.subplots(1,1)
fig.set_figwidth(15)
x = BATCH_SIZE
print(x)
print(testlosslist1)
axs.plot(x,testlosslist1,x,testacclist1,x,sensitivity1)
axs.legend(('loss','accuracy','sensitivity'))
axs.set_xlabel('batch')
axs.set_ylabel('variables')

"""4"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=4,kernel_size=1,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=2,kernel_size=1,padding="same",activation=tf.nn.relu,name='m1_conv2');
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
sens_op = tf.norm(tf.gradients(m1_loss,input_x))

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []
sensitivity1 = []


for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc, sen = session.run([m1_loss,m1_acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    sensitivity1.append(sen)
    msg = "batch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}, Sensitivisy:{5:>6.1%} "
    print(msg.format(i, trainloss, trainacc, testloss, testacc, sen))
    
fig,axs = plt.subplots(1,1)
fig.set_figwidth(15)
x = BATCH_SIZE
print(x)
print(testlosslist1)
axs.plot(x,testlosslist1,x,testacclist1,x,sensitivity1)
axs.legend(('loss','accuracy','sensitivity'))
axs.set_xlabel('batch')
axs.set_ylabel('variables')

"""5"""

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=25,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=80,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
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
sens_op = tf.norm(tf.gradients(m1_loss,input_x))

session = tf.Session()
session.run(tf.global_variables_initializer())

trainlosslist1 = []
trainacclist1 = []
testlosslist1 = []
testacclist1 = []
sensitivity1 = []


for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    trainlosslist1.append(trainloss)
    trainacclist1.append(trainacc)
    testloss, testacc, sen = session.run([m1_loss,m1_acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist1.append(testloss)
    testacclist1.append(testacc)
    sensitivity1.append(sen)
    msg = "batch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}, Sensitivisy:{5:>6.1%} "
    print(msg.format(i, trainloss, trainacc, testloss, testacc, sen))
    
fig,axs = plt.subplots(1,1)
fig.set_figwidth(15)
x = BATCH_SIZE
print(x)
print(testlosslist1)
axs.plot(x,testlosslist1,x,testacclist1,x,sensitivity1)
axs.legend(('loss','accuracy','sensitivity'))
axs.set_xlabel('batch')
axs.set_ylabel('variables')