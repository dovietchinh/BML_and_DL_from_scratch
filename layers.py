import tensorflow as tf
import numpy as np


class LinearRegression(tf.Module):
    def __init__(self, input_shape):
        """[y = wT @ x +b ]
        """
        initializer = tf.keras.initializers.GlorotNormal()
        self.W = tf.Variable(initial_value=initializer(
            shape=input_shape), dtype=tf.float32, trainable=True, name='Weights')
        self.b = tf.Variable(initial_value=initializer(
            shape=(1, 1)), dtype=tf.float32, trainable=True, name='bias')
        #self.x = tf.constant([[0]])

    def predict(self, x):
        out = tf.matmul(tf.transpose(self.W, perm=[1, 0]), x) + self.b
        return out

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def train(self, x, y):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(self.W)
            g.watch(self.b)
            print(g.watched_variables())
            out = tf.matmul(tf.transpose(self.W, perm=[1, 0]), x) + self.b
            loss = self.loss(y, out)
        gradients = g.gradient(loss, [self.W, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.W, self.b]))

    def fit(self, x, y):
        return self.train(x, y)








