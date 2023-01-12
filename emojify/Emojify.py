# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-04-25-10:34 上午
import numpy as np
from emoji_utils import *
import emoji
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(1)

def sentences_to_indices(X, word_to_index, max_len):
    """
    converts an array of sentences (strings) into an array of indices corresponding to words
    in the sentence. The output shape should be such that it can be given to 'Embedding()'.

    :param X: array of sentences of shape (m,max_len).
    :param word_to_index: a dictionary containing the each word mapped to its index.
    :param max_len: maximum number of words in a sentence.
    :return: X_indices --array of indices corresponding to words in the sentences X.
    """
    m = X.shape[0]
    X_indices = np.zeros(shape=(m, max_len))

    for i in range(m):
        sentences_words = X[i].lower().split()
        j=0;
        for w in sentences_words:
            X_indices[i,j] = word_to_index[w]
            j+=1

    return X_indices

def pretrain_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained Glove 50-dimensional vectors.

    :param word_to_vec_map: dictionary mapping words to their GloVe vector representation.
    :param word_to_index: dictionary mapping from words to their indices in vocabulary.
    :return: pretrained layer Keras instance
    """

    vocab_len = len(word_to_index)+1
    emb_dim = word_to_vec_map['cucumber'].shape[0]

    # 1. Initialize the embedding matrix as a numpy array of zeros
    emb_matrix = np.zeros(shape=(vocab_len, emb_dim))

    # 2. Set each row "idx" of the embedding matrix to be
    #    the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # 3. Define Keras embedding layer with the correct input and output sizes
    embedding_layer = layers.Embedding(vocab_len, emb_dim, trainable=True)

    # 4. Build the embedding layer, which is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))

    # 5. Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def Emojify_model(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify model's graph
    :param input_shape: shape of the input, usually (max_len,)
    :param word_to_vec_map: dictionary mapping every word in a vocabulary into its 50-dimensional vector representation.
    :param word_to_index:  dictionary mapping from words to their indices in the vocabulary (400,001 words).
    :return: a model instance in Keras
    """

    # 1. Define sentence_indices as the input of the graph.
    #    It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = keras.Input(shape=(input_shape), dtype='int32')

    # 2. Create the embedding layer pretrained with GloVe vectors
    embedding_layer = pretrain_embedding_layer(word_to_vec_map, word_to_index)

    # 3. Propagate sentence_indices through the embedding layer
    embeddings = embedding_layer(sentence_indices)

    # 4. Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = layers.LSTM(units=128, return_sequences=True, dropout=0.5)(embeddings)
    X = layers.LSTM(units=128, return_sequences=False, dropout=0.5)(X)
    X = layers.Dense(units=5)(X)
    X = layers.Activation('softmax')(X)

    # 5. Create Model instance which converts sentence_indices into X.
    model = keras.Model(inputs = sentence_indices, outputs=X)

    return model

if __name__=='__main__':
    # load the data
    X_train, Y_train = read_csv('data/train_emoji.csv')
    X_test, Y_test = read_csv('data/tesss.csv')
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    # find the sentence with the maximum length
    maxLen = len(max(X_train, key=len).split())

    model = Emojify_model((maxLen,), word_to_vec_map, word_to_index)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C=5)
    model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True)

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C=5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)

    print()
    print("Test accuracy = ", acc)
