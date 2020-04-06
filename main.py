import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing.image import ImageDataGenerator
from net import inference
import time

# Package 'tensorflow.examples.tutorials.mnist' requries manual download from
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials

# Parameters
num_pattrens = 15
image_size = 28
batch_size = 128
scale = 0.2
learning_rate = 0.001
n_epochs = 150

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
y_train = np_utils.to_categorical(y_train ,num_classes=10)
noise = 0

Mnist = input_data.read_data_sets('MNIST_data/', one_hot = True, validation_size = 0)

y = tf.placeholder(tf.float32,[None, 10])
x_image = tf.placeholder(tf.float32,[None, 28,28,1])

# applying a random rotation and a random shift to the training images
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.4)
datagen.fit(X_train)

outputs, parameters = inference(x_image,noise,image_size,batch_size,num_pattrens,scale)
prediction = tf.nn.softmax(outputs)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=outputs, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
init = tf.global_variables_initializer()
model_path = './modelParameters/dcnet_mnist.ckpt'

loss_summary = np.zeros(n_epochs)
accu_summary = np.zeros(n_epochs)

# Network training
start = time.clock()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = Mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = Mnist.train.next_batch(batch_size)
            X_batch = np.reshape(X_batch, [batch_size, image_size, image_size, 1])
            train_accuracy = accuracy.eval(feed_dict={x_image: X_batch, y: y_batch})
            sess.run(training_op, feed_dict={x_image: X_batch, y: y_batch})
            loss = sess.run(loss_op, feed_dict={x_image: X_batch, y: y_batch})
            if iteration == n_batches - 1:
                loss_summary[epoch] = loss
                accu_summary[epoch] = train_accuracy
                print('Raw Image Epoch: %d, iteration: %d, loss = %.4f' % (epoch, iteration, loss))
                print('train accuracy = %.4f' % train_accuracy)
    save_path = saver.save(sess, model_path)

    for epoch in range(n_epochs):
        batch = 0
        n_batches = X_train.shape[0]// batch_size
        print ('Epoch:', epoch)
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
            train_accuracy = accuracy.eval(feed_dict={x_image: x_batch, y: y_batch})
            sess.run(training_op, feed_dict={x_image: x_batch, y: y_batch})
            loss = sess.run(loss_op, feed_dict={x_image: x_batch, y: y_batch})
            if batch == n_batches-1:
                loss_summary[epoch] = loss
                accu_summary[epoch] = train_accuracy
                print('Image Aug Epoch: %d, iteration: %d, loss = %.4f' % (epoch, batch, loss))
                print('train accuracy = %.4f' % train_accuracy)
                break
            batch += 1
    save_path = saver.save(sess, model_path)
    print('Trained Model Saved.')

elapsed = (time.clock() - start)
print("Time used:",elapsed)
