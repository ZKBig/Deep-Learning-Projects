# -*- coding: utf-8 -*-
# @Description:
# @author: victor
# @create time: 2021-03-07-8:52 下午
from __future__ import print_function

import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
import pygame
import time

# play the music according to the file path.
def play_music(filepath):
    pygame.mixer.init()
    # load the music
    pygame.mixer.music.load(filepath)
    print("Play the music...")
    pygame.mixer.music.play(start=0.0)
    # set the playing time, otherwise, the music cannot be played, and it will load at a time.
    time.sleep(300)
    print("Finished.")
    pygame.mixer.music.stop()

# X, Y, n_values, indices_values = load_music_utils()
# print('number of training examples:', X.shape[0])
# print('Tx (length of sequence):', X.shape[1])
# print('total # of unique values:', n_values)
# print('shape of X:', X.shape)
# print('Shape of Y:', Y.shape)

if __name__=="__main__":
    play_music('./data/30s_trained_model.mp3')