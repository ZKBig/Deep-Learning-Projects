# -*- coding: utf-8 -*-
# @Description: a simple sigmoid neural network to distinguish cat images from non-cat images.
# @author: victor
# @create time: 2021-02-24-9:35 下午

# numpy is the fundamental package for scientific computing with Python.
# h5py is a common package to interact with a dataset that is stored on an H5 file.
# matplotlib is a famous library to plot graphs in Python.
# PIL and scipy are used here to test your model with your own picture at the end.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from dnn_test.lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# print(train_set_y)

# Example of a picture
# index = 27
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print ("y = " + str(train_set_y[0, index]) + ", it's a '" + classes[np.squeeze(train_set_y[0, index])].decode("utf-8") +
# "' picture.")

'''
I. The steps of preprocessing the data:
1) Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
2) Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
3) "Standardize" the data
'''
m_train = train_set_x_orig.shape[0] #number of training examples
m_test = test_set_x_orig.shape[0] #number of test examples
num_px = train_set_x_orig.shape[1] #= height = width of a training image

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_set_x shape: " + str(train_set_x_orig.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x shape: " + str(test_set_x_orig.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
'''
Reshape the training and test data sets so that images of size (num_px, num_px, 3)
are flattened into single vectors of shape (num_px ∗ num_px ∗ 3, 1).
A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b * c ∗ d, a) 
is to use: 
                    X_flatten = X.reshape(X.shape[0], -1).T 
'''
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
# print ("train_set_y shape: " + str(train_set_y.shape))
# print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
# print ("test_set_y shape: " + str(test_set_y.shape))
# print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

# standardize the dataset
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def preprocess_dataset():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    train_set_x_flatten = (train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T) / 255
    test_set_x_flatten = (test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T) / 255

    p={"m_train":m_train,
       "m_test":m_test,
       "num_px":num_px,
       "train_set_x_flatten":train_set_x_flatten,
       "test_set_x_flatten":test_set_x_flatten
       }

    return p
'''
II. General Architecture of the learning algorithm
1) Initialize the parameters of the model.
2) Learn the parameters for the model by minimizing the cost.
3) Use the learned parameters to make predictions (on the test set).
4) Analyse the results and conclude.

The main steps for building a Neural Network are:
Define the model structure (such as number of input features)
Initialize the model's parameters
Loop:
a. Calculate current loss (forward propagation)
b. Calculate current gradient (backward propagation)
c. Update parameters (gradient descent)
'''

# Activation function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# initializing parameters
"""
This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

Argument:
dim -- size of the w vector we want (or number of parameters in this case)

Returns:
w -- initialized vector of shape (dim, 1)
b -- initialized scalar (corresponds to the bias)
"""
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b

# Forward and Backward Propagation
"""
Implement the cost function and its gradient for the propagation explained above

Arguments:
w -- weights, a numpy array of size (num_px * num_px * 3, 1)
b -- bias, a scalar
X -- data of size (num_px * num_px * 3, number of examples)
Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

Return:
cost -- negative log-likelihood cost for logistic regression
dw -- gradient of the loss with respect to w, thus same shape as w
db -- gradient of the loss with respect to b, thus same shape as b

Tips:
- Write your code step by step for the propagation. np.log(), np.dot()
"""
def propagate(w, b, X, Y):
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * (np.dot(X, (A - Y).T))
    db = (1 / m) * (np.sum(A - Y))

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}
    return grads, cost

# Optimization
"""
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
"""
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# Predict
'''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
'''
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction

# Merge all functions into a model
"""
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
"""
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

def judge(image_name, d):
    image = np.array(ndimage.imread(image_name, flatten=False))
    image = image / 255.
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    train_set_x_flatten = (train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T) / 255
    test_set_x_flatten = (test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T) / 255

    # m_train, m_test, num_px, train_set_x_flatten, test_set_x_flatten = preprocess_dataset()

    index = 14
    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    plt.show()

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=20000,
              learning_rate=0.005, print_cost=True)

    if test_set_y[0, index]==1:
        print("This picture is a cat")
    else:
        print("This picture is not a cat")

    # Plot learning curve (with costs)
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()

    # learning_rates = [0.01, 0.001, 0.0001]
    # models = {}
    # for i in learning_rates:
    #     print("learning rate is: " + str(i))
    #     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
    #                            print_cost=False)
    #     print('\n' + "-------------------------------------------------------" + '\n')
    #
    # for i in learning_rates:
    #     plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
    #
    # plt.ylabel('cost')
    # plt.xlabel('iterations (hundreds)')
    #
    # legend = plt.legend(loc='upper center', shadow=True)
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # plt.show()
