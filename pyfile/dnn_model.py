import tensorflow as tf
import numpy as np
from scipy.io import loadmat


def init_parameters(layer_dims):
    '''
    initialize network parameters
    :param layer_dims: a list, number of neuron for each layer
    :return: a dict, contains weights and bias
    '''

    parameters = {}

    l = len(layer_dims)

    for i in range(l - 1):
        W = tf.compat.v1.get_variable('W'+str(i+1), [layer_dims[i+1], layer_dims[i]],
                                      initializer=tf.glorot_uniform_initializer())
        b = tf.compat.v1.get_variable('b'+str(i+1), [layer_dims[i+1], 1], initializer=tf.zeros_initializer())

        parameters['W'+str(i+1)] = W
        parameters['b'+str(i+1)] = b

    return parameters


def forward_propagation(X, parameters):
    '''
    forward propagation of neuron network
    :param X: input train data, shape of(features_number, number_of_data)
    :param parameters: the output of function init_parameters()
    :return: Z: the result of forward, propagation, caches contains middle output of each layer(A, Z)
    '''

    num_layers = len(parameters) // 2
    A = X

    for i in range(num_layers):
        Z = tf.add(tf.matmul(A, parameters['W'+str(i+1)]), parameters[''])


if __name__ == '__main__':

    data = loadmat('./datasets/features.mat').get('features')
    label = loadmat('./datasets/label.mat').get('label')

    classes = 3

    layer_dims = [data.shape[0], 10, 6, 3, classes]

    parameter = init_parameters(layer_dims=layer_dims)