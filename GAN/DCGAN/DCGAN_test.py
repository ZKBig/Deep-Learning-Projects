'''
Author: your name
Date: 2021-05-06 10:33:46
LastEditTime: 2021-05-06 11:49:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /undefined/Users/wangzheng/Desktop/Deep Learning/GAN/DCGAN/DCGAN_test.py
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from tensorflow.keras import layers

class DCGenerator(keras.Model):
    def __init__(self):
        super(DCGenerator, self).__init__()
        # input: [b, 100]=>[b, 3*3*512]
        self.fc = layers.Dense(3*3*512)
        # [b, 3*3*512] => [b, 3, 3, 512]
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()
        # [b, 3, 3, 512] => [b, 64, 64, 3]
        self.conv2 = layers.Conv2DTranspose(128, 5， 2， 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def model(self, inputs, training = None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.relu(x)

        x = tf.nn.relu(self.bn1(self.conv1(x), training = training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training = training))
        x = self.conv3(x)
        x = tf.tanh(x)

        return x
        


class DCDiscriminator(keras.Model):
    def __init__(self):
        super(DCDiscriminator, self).__init__()

        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')

        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def model(self, inputs, training = None):

        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training = training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training = training))

        x = self.flatten(x)

        out = self.fc(x)

        return out

if __name__ == '__main__':
    





