import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py

TRAIN_FILE = 'datasets/train_signs.h5'
TEST_FILE = 'datasets/test_signs.h5'

#HYPERPARAMETERS
LEARNING_RATE = 0.009
NUM_EPOCHS = 400
BATCH_SIZE = 64
N_ITERATIONS = int(1080/BATCH_SIZE)

PRINT_COST = True

def main():
    tf.compat.v1.disable_eager_execution()
    list_classes, train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()

    input_tensor = tf.compat.v1.placeholder("float", [None, 64, 64, 3])
    output_tensor = tf.compat.v1.placeholder("float", [None, 6])

    weights = {
        'W1': tf.compat.v1.get_variable("W1", [4, 4, 3, 8], initializer=tf.initializers.GlorotUniform()),
        'W2': tf.compat.v1.get_variable("W2", [2, 2, 8, 16], initializer=tf.initializers.GlorotUniform())
    }

    C1 = tf.nn.conv2d(input_tensor, weights['W1'], strides=[1,1,1,1], padding="SAME")
    A1 = tf.nn.relu(C1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding="SAME")

    C2 = tf.nn.conv2d(P1, weights['W2'], strides=[1,1,1,1], padding="SAME")
    A2 = tf.nn.relu(C2)
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding="SAME")

    F = tf.keras.layers.Flatten(P2)
    output_layer = tf.keras.layers.Dense(F, 6, activation = None)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output_layer, labels = output_tensor))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    costs = []
    for epoch in range(NUM_EPOCHS):
        minibatches = random_mini_batches(input_tensor, output_tensor)
        minibatch_cost = 0.
        for batch in minibatches:
            (minibatch_X, minibatch_Y) = batch

            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            _, temp_cost = sess.run([optimizer, cost], feed_dict={input_tensor: minibatch_X, output_tensor: minibatch_Y})

            minibatch_cost += temp_cost / N_ITERATIONS

        if PRINT_COST == True and epoch % 100 == 0:
            print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if PRINT_COST == True and epoch % 1 == 0:
            costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(LEARNING_RATE))
    plt.show()

    # Calculate the correct predictions
    predict_op = tf.argmax(output_layer, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(output_tensor, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({input_tensor: train_set_x, output_tensor: train_set_y})
    test_accuracy = accuracy.eval({input_tensor: test_set_x, output_tensor: test_set_y})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def load_dataset():
    train_signs = h5py.File(TRAIN_FILE, 'r')
    list_classes = np.array(train_signs['list_classes'])
    train_set_x = np.array(train_signs['train_set_x'])
    train_set_y = np.array(train_signs['train_set_y'])

    test_signs = h5py.File(TEST_FILE, 'r')
    test_set_x = np.array(test_signs['test_set_x'])
    test_set_y = np.array(test_signs['test_set_y'])

    return list_classes, train_set_x, train_set_y, test_set_x, test_set_y

if __name__ == '__main__':
    main()
