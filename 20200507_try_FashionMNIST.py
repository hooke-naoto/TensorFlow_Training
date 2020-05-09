# Install h5py
## "h5py" is a Python library which can handle HDF5 files.
## HDF is a Hierarchical Data Format and save NumPy data as binary file.
!pip install -q h5py pyyaml


from __future__ import absolute_import, division, print_function, unicode_literals

# Import
import os

# Import: TensorFlow, tf.keras
import tensorflow as tf
from tensorflow import keras

# Import: numpy, matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Show TensorFlow version
print(tf.__version__)

# Import "Fashion MNIST" and load to variables
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show train data and test data
print("train_images - shape: " + str(train_images.shape))
print("train_labels - length: " + str(len(train_labels)))
print("train_labels: " + str(train_labels))
print("test_images - shape: " + str(test_images.shape))
print("test_labels - length: " + str(len(test_labels)))
print("test_labels: " + str(test_labels))

# Show first 25 images from train data
# plt.figure(figsize=(5,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]], fontsize=6)
# plt.show()

# Show an image from train data
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Pre-processing for all images (Conver max 255 to max 1.)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define NN model layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # 1st layer: Same size as images.
    keras.layers.Dense(128, activation='relu'),     # 2nd layer
    keras.layers.Dense(10, activation='softmax')    # 3rd layer: Same size as labels.
])

# Compile NN model
model.compile(optimizer='adam',                         # Optimizer: How to optimize deviations while training.
              loss='sparse_categorical_crossentropy',   # Loss Function: How to evaluate accuracy of trained NN model.
              metrics=['accuracy'])                     # Metrics: How to monitor for training and test.

# Train NN model with train images/labels
model.fit(train_images, train_labels, epochs=5)

# Test NN model with test images/labels
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# Show accuracy results
print('Accuracy results with test data: ', test_acc)

# Predict test images with trained NN model
predictions = model.predict(test_images)

# Show a predicted result about the first data of test image.
## "predictions" contains confidences of each class nameself.
print(predictions)
## "np.argmax" shows index number in class_names of highest confidence.
print("[Results of the first image in test_images]")
print("True label: " + str(test_labels[0]) + " = " + str(class_names[test_labels[0]]))
print("Predicted label: " + str(np.argmax(predictions[0])) + " = " + str(class_names[np.argmax(predictions[0])]))

######## Functions for confidence graphs for all 10 classes ######## START
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color,
                                    fontsize=6)
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
######## Functions for confidence graphs for all 10 classes ######## END

# Show results of several test images with each predicted label and true label.
num_rows = 4
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Predict a selected test image with trained NN modelself.
## TensorFlow is for listed data therefor a single data should be converted to list.
img = test_images[0]
print("Shape of a selected image: " + str(img.shape))
img = (np.expand_dims(img,0))
print("Shape of a selected image converted as list: " + str(img.shape))
predictions_for_an_image = model.predict(img)
print("Predicted confidences: " + str(predictions_for_an_image))
## Plot a graph
plot_value_array(0, predictions_for_an_image, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print("Predicted label: " + class_names[np.argmax(predictions_for_an_image[0])])
