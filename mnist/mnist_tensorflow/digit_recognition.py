# DEFINITIONS
# tensor: An array, [width, height], containing numerical values. Can be 1D (Vector), 2D (Matrix), 3D (3-Tensor), ... n (N-Tensor)
# tensorflow.placeholder(): Creates a placeholder tensor that will be fed at run-time. Sort of like defining global variable that isn't defined until run time
# tensorflow.truncated_normal(): Generates tensor of random numbers from normal distribution specified by parameters. Tensor size is specified by shape parameter, stddev specifies the standard deviation of the normal distribution
# cross-entropy / log loss: A loss function that utilizes logs. If the predicted value is equal to the expected value (i.e. 7=7), then return -log(predict_val) = 1. If predicted value not equal to expected (i.e. predicted_val=6.9), then return -log(1-predicted_val). Higher loss for worse prediction
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# PLOT ARRAYS
iterations = []
acc = []

# NUMBER OF NODES PER LAYER
input_nodes = 784  # Each image is 28x28 pixels = 784 inputs
hidden1_nodes = 512
hidden2_nodes = 256
hidden3_nodes = 128
output_nodes = 10  # Only 10 digits possible

# HYPERPARAMETERS
learning_rate = 5e-4  # Rate at which weights are tuned to reduce loss function and converge to minimum. Higher learning rate will converge faster but can overshoot minimum
n_iterations = 1000 # Number of times we go through learning steps. The weights will be tuned n_iterations number of time
batch_size = 128  # Every training step will utilize 128 examples
dropout = 0.5  # Percent of nodes that are randomly shut off at each training step. By turning off nodes randomly, model can not just "memorize" training set, which prevents overfitting.

# LOADING MNIST DATASET
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# VARIABLES FOR TRAINING DATA
n_train = mnist.train.num_examples  # 55,000 for training
n_validation = mnist.validation.num_examples  # 5000 for validation
n_test = mnist.test.num_examples  # 10,000 for testing

# PLACEHOLDERS
x = tf.placeholder("float", [None, input_nodes])  # Creates a placeholder empty array of floats. The None specifies that we do not know how many images we will feed into the model, so the placeholder is an empty array of size [unspecified, 784]
y = tf.placeholder("float", [None, output_nodes])  # Placeholder tensor for where we will load the correct answers
keep_prob = tf.placeholder(tf.float32)  # Creates placeholder empty array of type float32, but since shape parameter is not specified as above, the placeholder can be fed a tensor of any size

# WEIGHTS
weights = {
    'w1': tf.Variable(tf.truncated_normal([input_nodes, hidden1_nodes], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([hidden1_nodes, hidden2_nodes], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([hidden2_nodes, hidden3_nodes], stddev=0.1)),
    'w4': tf.Variable(tf.truncated_normal([hidden3_nodes, output_nodes], stddev=0.1))
}

# BIASES
bias = {
    'b1': tf.Variable(tf.constant(0.1, shape=[hidden1_nodes])), # biases should always be instantiated with a small constant value so they contribute to propagation.
    'b2': tf.Variable(tf.constant(0.1, shape=[hidden2_nodes])), # tf.constant creates a tensor of size specified by the second parameter with all entries matching the first parameter
    'b3': tf.Variable(tf.constant(0.1, shape=[hidden3_nodes])),
    'out': tf.Variable(tf.constant(0.1, shape=[output_nodes]))
}

# MODEL LAYERS
layer_1 = tf.add(tf.matmul(x, weights['w1']), bias['b1']) # Creates a tensor by doing: matrix multiplication between the inputs to the first layer and the weights for those nodes, then adding those to the biases
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), bias['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), bias['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob) # Takes layer 3 tensor, and randomly deactivates keep_prob worth of nodes
output_layer = tf.matmul(layer_3, weights['w4']) + bias['out']

# LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer)) # Returns a tensor containing the mean of the cross-entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Uses the Adam Optimization method to minimize the cross-entropy tensor

# ACCURACY
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1)) # tf.argmax(output_layer, 1) finds the digit that the network sees as the most probable to match the inputted the image. A list of booleans representing whether the predictions were right is returned
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # tf.cast casts the input boolean tensor to floats. tf.reduce_mean calculates average number right

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        x: batch_x, y: batch_y, keep_prob: dropout
        })

    # print loss and accuracy (per minibatch)
    minibatch_loss, minibatch_accuracy = sess.run(
        [cross_entropy, accuracy],
        feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}
        )
    print(
        "Iteration",
        str(i),
        "\t| Loss =",
        str(minibatch_loss),
        "\t| Accuracy =",
        str(minibatch_accuracy)
        )
    iterations.append(i)
    acc.append(minibatch_accuracy)
plt.plot(iterations, acc)
plt.show()

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

digits={
    0 : "hand_drawn_images/zero.png",
    1 : "hand_drawn_images/one.png",
    2 : "hand_drawn_images/two.png",
    3 : "hand_drawn_images/three.png",
    4 : "hand_drawn_images/four.png",
    5 : "hand_drawn_images/five.png",
    6 : "hand_drawn_images/six.png",
    7 : "hand_drawn_images/seven.png",
    8 : "hand_drawn_images/eight.png",
    9 : "hand_drawn_images/nine.png"
}
for r in range(0, 10, 1):
    img = np.invert(Image.open(digits[r]).convert('L')).ravel()
    prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={x: [img]})
    print("Prediction for test image:", np.squeeze(prediction))

