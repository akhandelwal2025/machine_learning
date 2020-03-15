import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def main():
    data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    train_images = train_images/255.0
    test_images = test_images/255.0

    model = keras.Sequential() #Creates a new model that will execute layers in sequence
    model.add(keras.layers.Flatten(input_shape=(28,28))) #Flatten layer takes input tensor (28x28 pixeled image) and breaks it down into a tensor of size (1x784)
    model.add(keras.layers.Dense(128, activation="relu")) #Creates a fully-connected hidden layer with 128 nodes. Relu Activation function just sets negative numbers to 0 and increases positive numbers by certain gain
    model.add(keras.layers.Dense(10)) #Output layer has 10 nodes to indicate which of the 10 categories the input picture is. Each node will output a percentage, with the highest one being the model's prediction.
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10) #Train the model for 10 epochs with the training data/labels
    #test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) #Test accuracy of model with bunch of test cases
    plt.figure(figsize=(10, 10))
    plt.imshow(test_images[7], cmap=plt.cm.binary)
    plt.show()
    prediction = model.predict(np.array([test_images[7]]))
    print(class_names[np.argmax(prediction)])
    #print('\nTest accuracy:', test_acc)

def display_first_25():
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

if __name__ == '__main__':
    main()