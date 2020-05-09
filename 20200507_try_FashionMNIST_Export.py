# Install h5py before this program: pip install -q h5py pyyaml
## "h5py" is a Python library which can handle HDF5 files (xxxx.h5).
## HDF is a Hierarchical Data Format and save NumPy data as binary file.


######## Import ######## START
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
# Import: TensorFlow, tf.keras
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version: " + tf.__version__) # Show TensorFlow version
# Import: numpy, matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt
print("\n")
######## Import ######## START


######## Import "Fashion MNIST" ######## START
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
######## Import "Fashion MNIST" ######## END


######## Preprocessing for all images (max 255 -> max 1) ######## START
train_images = train_images / 255.0
test_images = test_images / 255.0
######## Preprocessing for all images (max 255 -> max 1) ######## END


######## Define NN model ######## START
print("######## Define NN model ######## START")
def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),     # 1st layer: Same size as images.
        keras.layers.Dense(128, activation='relu'),     # 2nd layer
        keras.layers.Dense(10, activation='softmax')    # 3rd layer: Same size as labels.
    ])
    model.compile(optimizer='adam',                         # Optimizer: How to optimize deviations while training.
                  loss='sparse_categorical_crossentropy',   # Loss Function: How to evaluate accuracy of trained NN model.
                  metrics=['accuracy'])                     # Metrics: How to monitor for training and test.
    return model
model = create_model()
model.summary()
print("\n")
######## Define NN model ######## END


######## Define Checkpoint Callback & Train NN model ######## START
# # Please ignore warning about optimizer because of just information to avoid old methods.
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, save_weights_only=True, verbose=1)
# model = create_model()
# model.fit(train_images, train_labels,
#           epochs = 10,
#           validation_data = (test_images,test_labels),
#           callbacks = [cp_callback] # Give Callback while training.
#           )
######## Define Checkpoint Callback & Train NN model ######## END


######## Evaluate an untrained model ######## START
# # Create an untrained NN model and evaluate it.
# # It will be a bad result such around 10% accuracy.
# model = create_model()
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
######## Evaluate an untrained model ######## END


######## Evaluate a restored NN model ######## START
# # Load weights from Checkpoint data to the untrained NN model and evaluate it.
# # BE CAREFUL: NN model and weights should have same structure for this process.
# # It will be a good result such around 80-90% accuracy.
# model.load_weights(checkpoint_path) # Load weights from latest Checkpoint.
# loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
######## Evaluate a restored NN model ######## END


######## Trials for customized Checkpoint ######## START
# Checkpoint can be customized: file name for each epoch, frequency of Checkpoint, etc.
# Following example shows file name for each fixed epochs.
print("######## Trials for customized Checkpoint ######## START")
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt" # {epoch:04d} = the epoch number as 4-digit with string format.
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, save_weights_only=True, verbose=1, period=2) # "period" defines saving weights each fixed epochs.
model = create_model()
model.fit(train_images, train_labels,
          epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [cp_callback], # Give Callback while training.
          # verbose=0 # verbose=0 means no log showing for this operation.
          )
print("\n")
######## Trials for customized Checkpoint ######## END


######## Load latest Checkpoint ######## START
print("######## Load latest Checkpoint ######## START")
latest = tf.train.latest_checkpoint(checkpoint_dir) # Select latest Checkpoint.
model = create_model()
model.load_weights(latest) # Load weights from latest Checkpoint.
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("\n")
######## Load latest Checkpoint ######## END


######## [Option] Weights can be saved manually and loaded ######## START
# # Save manually.
# model.save_weights('./checkpoints/my_checkpoint')
# # Create an untrained NN model. Load manually.
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')
# loss,acc = model.evaluate(test_images, test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print("\n")
######## [Option] Weights can be saved manually and loaded ######## END


######## Save the entire NN model with weights ######## START
print("######## Save the entire NN model with weights ######## START")
# Save NN model.
model = create_model()
model.fit(train_images, train_labels, epochs=5)
model.save('my_model.h5')
# Load NN model.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
# Evaluate loaded NN model.
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
print("\n")
######## Save the entire NN model with weights ######## END
