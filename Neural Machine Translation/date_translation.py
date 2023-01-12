# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-04-25-2:51 下午

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as k
import numpy as np

from faker import Faker
from tqdm import tqdm
from nmt_utils import *
import matplotlib.pyplot as plt

n_a = 32
n_s = 64
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# Defined shared layers as global variables
repeator = layers.RepeatVector(Tx)
concatenator = layers.Concatenate(axis=-1)
densor1 = layers.Dense(10, activation='tanh')
densor2 = layers.Dense(1, activation='relu')
activator = layers.Activation(softmax, name='attention_weights')
dotor = layers.Dot(axes=1)
post_activation_LSTM_cell = layers.LSTM(n_s, return_state=True)
output_layer = layers.Dense(len(machine_vocab),activation=softmax)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a product of the attention weights
    'alphas' and the hidden states 'a' in the Bi_LSTM.

    :param a: hidden state output of the Bi-LSTM, numpy array of shape (m, Tx, 2*n_a)
    :param s_prev: previous hidden state of the (post-attention) LSTM, numpy_array of shape (m, n_s)
    :return: context vector, input of the next (post-attention) LSTM cell
    """
    # 1. Repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev)

    # 2. Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])

    # 3. Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(concat)

    # 4. Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)

    # 5. Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)

    # 6.  Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])

    return context

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    :param Tx: length of the input sequence
    :param Ty: length of the output sequence
    :param n_a: hidden state size of the Bi-LSTM
    :param n_s: hidden state size of the post-attention LSTM
    :param human_vocab_size: size of the python dictionary "human_vocab"
    :param machine_vocab_size:  size of the python dictionary "machine_vocab"
    :return: Keras model instance
    """
    X = keras.Input(shape=(Tx, human_vocab_size))
    s0 = keras.Input(shape=(n_s,), name='s0')
    c0 = keras.Input(shape=(n_s,), name='c0')
    s = s0
    c= c0
    outputs = []

    # Step 1: Define the pre-attention Bi-LSTM.
    a = layers.Bidirectional(layers.LSTM(units=n_a, return_sequence=True))(X)

    # step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(a,s)
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)
        # Step 2.D: Append "out" to the "outputs" list
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs
    model = keras.Model(inputs=[X, s0, c0], outputs=outputs)

    return model

if __name__=='__main__':
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.summary()

    opt = keras.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    outputs = list(Yoh.swapaxes(0, 1))

    model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
    model.load_weights('models/model.h5')

    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
        prediction = model.predict([source, s0, c0])
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]

        print("source:", example)
        print("output:", ''.join(output), "\n")


