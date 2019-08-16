import numpy as np
import matplotlib.pyplot as plt
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
