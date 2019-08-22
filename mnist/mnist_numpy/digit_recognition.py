import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
input_nodes = 784
hidden1_layer = 512
hidden2_layer = 256
hidden3_layer = 128
output_nodes = 10

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_train = mnist.train.num_exmaples
n_validation = mnist.validation.num_examples
n_test = mnist.text.num_exmaples

weights = {}
bias = {}

def set_parameters():
    weights: {
        'w1': create_weights(input_nodes, hidden1_layer),
        'w2': create_weights(hidden1_layer, hidden2_layer),
        'w3': create_weights(hidden2_layer, hidden3_layer),
        'w4': create_weights(hidden3_layer, output_nodes)
    }

    bias: {
        'b1': create_bias(hidden1_layer, 0.1),
        'b2': create_bias(hidden2_layer, 0.1),
        'b3': create_bias(hidden3_layer, 0.1),
        'out': create_bias(output_nodes, 0.1)
    }


def create_weights(first_layer, second_layer):
    return np.random.normal(0, 0.1, [first_layer, second_layer])

def create_bias(layer_count, constant_val):
    return np.full(shape=layer_count, fill_value=constant_val)

def run_session(input_arr):
    layer_1 = np.add(np.matmul(weights['w1'], input_arr), bias['b1'])
    layer_2 = np.add(np.matmul(weights['w2'], layer_1), bias['b2'])
    layer_3 = np.add(np.matmul(weights['w3'], layer_2), bias['b3'])
    layer_3 = dropout(layer_3)
    output_layer = np.add(np.matmul(weights['w4'], layer_3), bias['out'])

def dropout(layer):
    num_rows, num_cols = layer.shape()
    for row in range(num_rows):
        for col in range(num_cols):
            if(random.random() < dropout):
                layer[row][col] = 0
    return (1/dropout) * layer