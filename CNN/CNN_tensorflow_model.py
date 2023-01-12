# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-02-28-6:14 下午
import math
import matplotlib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
import os

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

np.random.seed(1)

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
# index = 6
# plt.imshow(X_train_orig[index])
# plt.show()
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T


# GRADED FUNCTION: create_placeholders
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
    Y = tf.placeholder(tf.float32, shape=(None, n_y), name="Y")

    return X, Y


# GRADED FUNCTION: initialize_parameters
'''
Implement the function below to initialize the parameters in tensorflow. 
You are going use Xavier Initialization for weights and Zero Initialization for biases.
 The shapes are given below. As an example, to help you, for W1 and b1 you could use:
 
W1 = tf.get_variable("W1", [25,12288], initializer = tf.truncated_normal_initializer(stddev=0.1))
b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())

'''
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Note that we will hard code the shape values in the function to make the grading simpler.
    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.compat.v1.set_random_seed(1)  # so that your "random" numbers match ours

    W1 = tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.compat.v1.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1, "W2": W2}

    return parameters


# GRADED FUNCTION: forward_propagation
'''
In TensorFlow, there are built-in functions that implement the convolution steps for you.

1) tf.nn.conv2d(X,W, strides = [1,s,s,1], padding = 'SAME'): 
   given an input X and a group of filters W, this function convolves W's filters on X. 
   The third parameter ([1,s,s,1]) represents the strides for each dimension of the input 
   (m, n_H_prev, n_W_prev, n_C_prev). 
   Normally, you'll choose a stride of 1 for the number of examples (the first value) and 
   for the channels (the fourth value), which is why we wrote the value as [1,s,s,1]. 
   
2) tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME'):
   given an input A, this function uses a window of size (f, f) and strides of size (s, s) 
   to carry out max pooling over each window. For max pooling, 
   it is usually operated on a single example at a time and a single channel at a time. 
   So the first and fourth value in [1,f,f,1] are both 1. 
   
3) tf.nn.relu(Z):
   computes the elementwise ReLU of Z (which can be any shape). 
   
4) tf.contrib.layers.flatten(P): 
   given a tensor "P", this function takes each training (or test) 
   example in the batch and flattens it into a 1D vector.
   If a tensor P has the shape (m,h,w,c), where m is the number of examples (the batch size), 
   it returns a flattened tensor with shape (batch_size, k), where  k=h×w×c.
   
5) tf.contrib.layers.fully_connected(F, num_outputs): 
   given the flattened input F, it returns the output computed using a fully connected layer. 
   The fully connected layer automatically initializes weights in the graph and 
   keeps on training them as you train the model.
   Hence, there is no need initializing those weights when initializing the parameters.
'''

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAX POOLING -> FLATTEN -> FULLY CONNECTED

    Note that for simplicity and grading purposes, we'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    # RELU
    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')

    # RELU
    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    # FLATTEN
    F = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(F, 6, activation_fn=None)

    return Z3


# GRADED FUNCTION: compute_cost
'''
1) tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y): 
   computes the softmax entropy loss. This function both computes the softmax activation function 
   as well as the resulting loss. 
2) tf.reduce_mean(P): computes the mean of elements across dimensions of a tensor P.
   Use this to calculate the sum of the losses over all the examples to get the overall cost. 
'''

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))

    return cost

# GRADED FUNCTION: model
'''
1)  tf.train.AdamOptimizer(learning_rate = ...).minimize(loss=...):
    Adam Optimizer
2)  tf.train.GradientDescentOptimizer(learning_rate = ...).minimize(cost=...)
    Gradient Descent Optimizer
3)  random_mini_batches(X, Y, mini_batch_size = 64, seed = 0)
4)  output_for_var1, output_for_var2 = sess.run(fetches=[var1, var2], 
    feed_dict={var_inputs: the_batch_of_inputs, var_labels: the_batch_of_labels}):
    
'''
def cnn_model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=1000, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAX POOL -> FLATTEN -> FULLY CONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.compat.v1.global_variables_initializer()

    # plt.figure(1)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.xticks(np.arange(0, num_epochs, 50))

    # Start the session to compute the tensorflow graph
    with tf.compat.v1.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                """
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost.
                # The feedict should contain a minibatch for (X,Y).
                """
                # Note When coding, we often use _ as a "throwaway" variable to store values
                # that we won't need to use later.
                # Here, _ takes on the evaluated value of optimizer, which we don't need
                # (and temp_cost takes the value of the cost variable).
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                # plot_duration(epoch, minibatch_cost, learning_rate, num_epochs)
                # plt.plot(epoch, minibatch_cost, '.')
                # plt.show()
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.compat.v1.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

def plot_duration(i, y1, learning_rate,num_iterations):
    plt.figure(1)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.xticks(np.arange(0,num_iterations,50))
    plt.plot(i, y1, '.')
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.disable_eager_execution()
    _, _, parameters = cnn_model(X_train, Y_train, X_test, Y_test)