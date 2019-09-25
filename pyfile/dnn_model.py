import os
import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import train_test_split
from tensorflow.python import ops


def init_parameters(layer_dims):
    """
    initialize network parameters
    :param layer_dims: a list, number of neuron for each layer
    :return: a dict, contains weights and bias
    """

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
    :return: Z: the result of forward propagation
    '''

    num_layers = len(parameters) // 2
    A = X

    for i in range(num_layers):
        Z = tf.add(tf.matmul(A, parameters['W'+str(i+1)]), parameters['b'+str(i+1)])
        A = tf.nn.relu(Z)

    return Z


def compute_l2_loss(parameters):
    """
    compute the loss of l2-regularization
    :param parameters: the network parameters include weights and bias
    :return: l2_loss
    """

    L = len(parameters) // 2
    l2_loss = 0

    for i in range(L):
        l2_loss += tf.nn.l2_loss(parameters['W'+str(i+1)])

    return l2_loss


def compute_cost(Z, Y, beta, l2_loss):
    """
    compute the network loss
    :param Z:  the output of forward_propagation
    :param Y: the true label
    :param beta: the hyper-parameters of l2_loss
    :param l2_loss: the output of compute_l2_loss
    :return: loss
    """

    logits = tf.compat.v1.transpose(Z)
    labels = tf.compat.v1.transpose(Y)

    cost = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)) + beta*l2_loss

    return cost


def mini_batch(X, Y, batch_size):
    """
    extract number of batch_size samples
    :param X: train data with shape of (number_features, number_samples)
    :param Y: train label with shape of (number_class, number_samples), which is one_hot encode
    :param batch_size:
    :return: a list, which each element consist of (batch_x, batch_y)
    """
    m = X.shape[0]  # number of training samples
    batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_x = X[:, permutation]
    shuffled_y = Y[:, permutation]

    num_batches = np.floor(m / batch_size)
    for k in range(0, num_batches):
        batch_x = shuffled_x[:, k*batch_size: k*batch_size+batch_size]
        batch_y = shuffled_y[:, k*batch_size: k*batch_size+batch_size]
        batches.append((batch_x, batch_y))

    if m % batch_size != 0:
        batch_x = shuffled_x[:, num_batches * batch_size: m]
        batch_y = shuffled_y[:, num_batches * batch_size: m]
        batches.append((batch_x, batch_y))

    return batches


def model(layer_dims, train_X, train_Y, test_X, test_Y, beta=0,learning_rate=0.0001, epochs=10000, batch_size=32, print_cost=True, plot_cost=True):

    ops.reset_default_graph()

    num_features, num_samples = train_X.shape
    print('train samples: ', num_samples)
    print('train features: ', num_features)

    X = tf.compat.v1.placeholder(tf.float32, shape=[num_features, None], name='input_X')
    Y = tf.compat.v1.placeholder(tf.float32, shape=[train_Y.shape[0], None], name='output_Y')

    parameters = init_parameters(layer_dims)
    l2_loss = compute_l2_loss(parameters)

    Y_hat = forward_propagation(train_X, parameters)
    cost = compute_cost(Z=Y_hat, beta=beta, Y=train_Y, l2_loss=l2_loss)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step = optimizer.minimize(cost)

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        costs = []
        for epoch in range(epochs):
            epoch_cost = 0.0
            learning_rate = 1.0 / (1+epoch) * learning_rate

            num_batches = int(num_samples / batch_size)
            batches = mini_batch(train_X, train_Y, batch_size)

            for batch in batches:
                batch_x, batch_y = batch
                _, batch_cost = sess.run([train_step, cost], feed_dict={X: batch_x, Y: batch_y})
                epoch_cost += batch_cost / num_batches

            if print_cost and epoch % 100 == 0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if plot_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

            pred = tf.equal(tf.argmax(Y_hat), tf.argmax(train_Y))
            accuracy = tf.reduce_mean(tf.cast(pred), float)
            print('Train accuracy after epoch %i: %.2f' % (epoch, accuracy.eval({X: train_X, Y: train_Y})))
            print('Test accuracy after epoch %i: %.2f' % (epoch, accuracy.eval({X: test_X, Y: test_Y})))

        if plot_cost:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per fives)')
            plt.title('Learning rate='+str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)

    return  parameters


def get_set_data(dir_path):
    """
    read files from a dir
    :param dir_path: should be like ('./datasets/train_set/')
    :return: a matrix which contains all data read from csv file
    """
    train_set_data = []
    files = os.listdir(dir_path)
    for file in files:
        if not os.path.isdir(file):
            file_name = os.path.basename(file)
            file_path = os.path.join(dir_path, file_name)

            with open(file_path, 'r') as fr:
                file_data = pd.read_csv(fr)
                file_data = np.array(file_data.get_values(), dtype=np.float32)

                print('fileNameL ', file_name, ' shape of file date:', file_data.shape)
                train_set_data.extend(file_data)

    train_set_data = np.array(train_set_data)
    print('shape of total train set data:', train_set_data.shape)

    return train_set_data


def get_train_test_from_csv(dir_path):
    """
    read data form of csv
    :param dir_path: the dataset path
    :return:train_x, train_y, test_x, test_y
    """
    train_path = os.path.join(dir_path, 'train_set')
    test_path = os.path.join(dir_path, 'test_set')

    train_data = get_set_data(train_path)
    test_data = get_set_data(test_path)

    train_x = train_data[, : -2]
    train_y = train_data[:, -1]

    test_x = test_data[, : -2]
    test_y = test_data[:, -1]

    return train_x.T, train_y, test_x.T, test_y


def get_train_test_from_mat(dir_path, feature_name, label_name, test_size):
    """
    read data, form of .mat
    :param dir_path: the dir path of .mat file
    :param feature_name: the features name(note train file name should be the same as key
    :param label_name: the label file name(note label file name should be the same as key
    :param test_size: the size of test data set
    :return: train_x, train_y, test_x, test_y
    """
    feature_path = os.path.join(dir_path, feature_name)
    feature_path = feature_path + '.mat'
    train_data = loadmat(feature_path).get(feature_name)

    label_path = os.path.join(dir_path, label_name)
    label_path = label_path + '.mat'
    label_data = loadmat(label_path).get(label_name)

    train_x, test_x, train_y, test_y = train_test_split(train_data, label_data, test_size=test_size)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':

    # data = loadmat('./datasets/feature.mat').get('feature')
    # label = loadmat('./datasets/label.mat').get('label')

    classes = 3

    layer_dims = [data.shape[0], 10, 6, 3, classes]
