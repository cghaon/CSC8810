# -*- coding: utf-8 -*-
"""HW3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XGvG7PYHZeqUoSxdh3u4Kq3o4XqSqWEP
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2
epoch = 100

#MNIST
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

#CIFAR
bar_fig = plt.figure(figsize=[10,5])
unique, counts = np.unique(np.argmax(data.train.labels,1), return_counts=True)
plt.bar(unique,counts)
plt.title("Data Distribution Before Data Augmentation")
plt.xticks(unique,np.arange(10));

tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)



# layer_conv1
net = tf.layers.conv2d(inputs=input_x, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

# layer_conv2
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

print(net)

net = tf.layers.flatten(net)

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
print(logits)

#
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss = tf.reduce_mean(cross_entropy)

softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));

opt = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer = opt.minimize(loss)

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')
print(weights_conv1)
print(weights_conv2)

weights_fc1 = get_weights_variable(layer_name='layer_fc1')
weights_fc_out = get_weights_variable(layer_name='layer_fc_out')
print(weights_fc1)
print(weights_fc_out)

"""Run"""

trainable_var_list = tf.trainable_variables()

session = tf.Session()
session.run(tf.global_variables_initializer())
test_batch_size = 256
train_batch_size = 64

grads = tf.gradients(loss, weights_fc_out)[0]
grads_norm = tf.norm(grads)
print(grads)
print(grads_norm)
hessian = tf.reduce_sum(tf.hessians(loss, weights_fc_out)[0], axis = 2)
print(hessian)
train_batch_size = 64



total_iterations = 0
fc1_shape = weights_fc1.get_shape().as_list()
fc1_shape.insert(0,0)
fc_out_shape = weights_fc_out.get_shape().as_list()
fc_out_shape.insert(0,0)
total_weights_fc1 = np.empty(fc1_shape,dtype=np.float32)
total_weights_fc_out = np.empty(fc_out_shape,dtype=np.float32)
gradslist = []
losslist = []
acclist = []
def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations
    global losslist
    global acclist
    global total_weights_fc1
    global total_weights_fc_out
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 10 == 0:
            # Calculate the accuracy on the training-set.
            los, acc = session.run([loss, accuracy], feed_dict=feed_dict_train)
            losslist.append(los)
            acclist.append(acc)
            grads_vals, hess_vals = session.run([grads, hessian], feed_dict=feed_dict_train)
            grads_norm_vals, grads_vals, hess_vals = session.run([grads_norm, grads, hessian], feed_dict={x: x_batch,y: y_true_batch})
            gradslist.append(grads_norm_vals)
            w_fc1,w_fc_out = session.run([weights_fc1,weights_fc_out])
            total_weights_fc1 = np.append(total_weights_fc1,[w_fc1],axis=0)
            total_weights_fc_out = np.append(total_weights_fc_out,[w_fc_out],axis=0)
            print(grads_vals.shape)
            print(hess_vals.shape)

            # Message for printing.
            msg = "Iteration: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}"

            # Print it.
            print(msg.format(i + 1, los, acc))
            

    # Update the total number of iterations performed.
    total_iterations += num_iterations

x_train = data.train.images
y_train = data.train.labels
np.random.shuffle(y_train) #randomlu shuffle labels
y_train.shape

np.arange(0,100,20)

def createBatches(train_x,train_y,batch_size):
    mini_batches = []
    data_num = train_x.shape[0]
    i = np.arange(data_num)
    np.random.shuffle(i)
    train_x = train_x[i]
    train_y = train_y[i]
    for i in range(0,data_num-batch_size,batch_size):
        x = train_x[i:i+batch_size]
        y = train_y[i:i+batch_size]
        mini_batches.append((x,y))
    if data_num % batch_size != 0:
        x = train_x[i+batch_size:data_num]
        y = train_y[i+batch_size:data_num]
        mini_batches.append((x,y))
    return mini_batches

sess = tf.Session() 
sess.run(tf.global_variables_initializer())         
BATCH_SIZE = 64
trainlosslist = []
trainacclist = []
testlosslist = []
testacclist = []
for i in range(epoch):
    batches = createBatches(x_train,y_train,BATCH_SIZE)
    for batch in batches:
        x_batch, y_true_batch = batch
        session.run(optimizer, feed_dict={x: x_batch,y: y_true_batch})
    trainloss, trainacc = session.run([loss,accuracy], feed_dict={x: x_batch,y: y_true_batch})
    trainlosslist.append(trainloss)
    trainacclist.append(trainacc)
    testloss, testacc = session.run([loss,accuracy],feed_dict={x:data.test.images,y:data.test.labels})
    testlosslist.append(testloss)
    testacclist.append(testacc)
    if i%5 == 0:
        print("Epoch: ",i,"Train Loss: ",trainloss,"Test Loss: ",testloss,"Test acc: ",testacc)

print(total_iterations)
print(len(gradslist))
print(len(testlosslist))
x = np.arange(0,epoch)
plt.plot(x,trainlosslist,x,testlosslist)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(('training','test'))
plt.pause(0.1)

data.test.cls = np.argmax(data.test.labels,axis=1)
def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(pred_op, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

print_test_accuracy()