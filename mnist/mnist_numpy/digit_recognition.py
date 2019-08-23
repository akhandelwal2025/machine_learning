import numpy as np
import matplotlib.pyplot as plt
import random
import gzip
from tensorflow.examples.tutorials.mnist import input_data

train_img_filepath = 'MNIST_data/train-images-idx3-ubyte.gz'
train_labels_filepath = 'MNIST_data/train-labels-idx1-ubyte.gz'
test_img_filepath = 'MNIST_data/t10k-images-idx3-ubyte.gz'
test_labels_filepath = 'MNIST_data/t10k-labels-idx1-ubyte.gz'

train_img, train_labels = []
image_size = 28
input_nodes = 784
hidden1_layer = 512
hidden2_layer = 256
hidden3_layer = 128
output_nodes = 10


learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

weights = {}
bias = {}


def main():
    set_parameters()
    for epoch in range(n_iterations):
        current_img, current_labels = get_next_train_batch(epoch)
        output_results = [] # predictions
        for img in current_img:
            output_results.append(run_session(img))
        calculate_cross_entropy()




def set_parameters():
    train_img, train_labels = get_train_img_labels()

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
    return output_layer

def dropout(layer):
    num_rows, num_cols = layer.shape
    for row in range(num_rows):
        for col in range(num_cols):
            if(random.random() < dropout):
                layer[row][col] = 0
    return (1/dropout) * layer

def get_train_img_labels():
    img_file = gzip.open(train_img_filepath, 'r')
    buffer = img_file.read(image_size * image_size * batch_size)
    img = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    img = img.reshape(batch_size, image_size, image_size, 1)

    label_file = gzip.open(train_labels_filepath, 'r')
    label = label_file.read()
    return img, label

def get_next_train_batch(epoch_num):
    img_return, label_return = []
    for i in range(epoch_num*batch_size, epoch_num*batch_size+batch_size, 1):
        img_return.append(np.asarray(train_img[i]).squeeze())

    for l in range(batch_size):
        buffer = train_labels.read(1)
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
        label_return.append(labels)

    return img_return, label_return

def calculate_cross_entropy(predictions, correct_labels):
    



