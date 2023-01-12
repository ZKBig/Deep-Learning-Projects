# -*- coding: utf-8 -*-
# @Description: Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that.
# @author: victor
# @create time: 2021-03-01-2:46 下午

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
import imageio
import os

# GRADED FUNCTION: compute_content_cost
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_W * n_H, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_W * n_H, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = (1 / (4 * n_W * n_H * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


# GRADED FUNCTION: gram_matrix
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))

    return GA

# Compute the style cost for a single layer.
'''
1) Retrieve dimensions from the hidden layer activations a_G;
2) Unroll the hidden layer activations a_S and a_G into 2D matrices;
3) Compute the Style matrix of the images S and G;
4) Compute the Style cost.
'''
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(a_S, shape=[n_H * n_W, n_C])
    a_G = tf.reshape(a_G, shape=[n_H * n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    #     J_style_layer = 1./(4 * n_C * n_C * n_H * n_W * n_H * n_W) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * tf.to_float(tf.square(n_C * n_H * n_W)))

    return J_style_layer

# compute style cost
'''
1) Select the activation (the output tensor) of the current layer.
2) Get the style of the style image "S" from the current layer.
3) Get the style of the generated image "G" from the current layer.
4) Compute the "style cost" for the current layer
5) Add the weighted style cost to the overall style cost (J_style)
'''
def compute_style_cost(model, sess, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        # In the inner-loop of the for-loop above, a_G is a tensor and hasn't been evaluated yet.
        # It will be evaluated and updated at each iteration when we run the TensorFlow graph in model_nn() below.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


# GRADED FUNCTION: total_cost
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style

    return J

# Solving the optimization problem
'''
1) Create an Interactive Session
2) Load the content image
3) Load the style image
4) Randomly initialize the image to be generated
5) Load the VGG19 model
6) Build the TensorFlow graph:
    Run the content image through the VGG19 model and compute the content cost
    Run the style image through the VGG19 model and compute the style cost
    Compute the total cost
    Define the optimizer and the learning rate
7) Initialize the TensorFlow graph and run it for a large number of iterations, 
   updating the generated image at every step.
'''

# Start the interactive session.
'''
Unlike a regular session, the "Interactive Session" installs itself as the default session to build a graph.
This allows you to run variables without constantly needing to refer to the session object (calling "sess.run()"), 
which simplifies the code.
'''
def construct_interactive_session():
    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    return sess

'''
a. Implement the model_nn() function.
b. The function initializes the variables of the tensorflow graph,
c. assigns the input image (initial generated image) as the input of the VGG19 model
d. and runs the train_step tensor (it was created in the code above this function) for a large number of steps.
'''
def model_nn(sess, input_image, content_image, style_image, STYLE_LAYERS,
             model, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, sess, STYLE_LAYERS)

    # compute the total cost
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step
    train_step = optimizer.minimize(J)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    sess = construct_interactive_session()
    sess.run(tf.global_variables_initializer())
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]

    # load the content image
    content_image = imageio.imread("images/louvre_small.jpg")
    content_image = reshape_and_normalize_image(content_image)

    # load the style image
    style_image = imageio.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)

    # generate the noise image
    generated_image = generate_noise_image(content_image)
    imshow(generated_image[0]);

    # load the vgg model
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

    model_nn(sess, generated_image, content_image, style_image, STYLE_LAYERS, model)