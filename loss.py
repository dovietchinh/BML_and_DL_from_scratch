import tensorflow as tf 
import numpy as np 

def mean_square_error(y_true,y_pred):
    return tf.square(tf.sum(y_true -y_pred))

def mean_absolute_error(y_true,y_pred):
    return tf.sum(tf.abs(y_true - y_pred))

def binary_crossentropy(y_true,y_pred):
    return -y_true*tf.log(y_pred) + (1-y_true)*tf.log(1-y_pred)

def categorical_crossentropy(y_true,y_pred):
    arg_max = tf.math.argmax(y_pred)
    return -y_true[arg_max]*y_pred[arg_max]
