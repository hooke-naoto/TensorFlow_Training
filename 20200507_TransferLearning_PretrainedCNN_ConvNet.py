######## Import ######## START
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
######## Import ######## END


######## Datasets ######## START
(raw_train, raw_validation, raw_test), metadata = tfds.load(
                                                            'cats_vs_dogs',
                                                            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                            with_info=True,
                                                            as_supervised=True,
                                                            )
# print(raw_train)
# print(raw_validation)
# print(raw_test)
######## Datasets ######## END


######## Check several images ######## START
# get_label_name = metadata.features['label'].int2str
# for image, label in raw_train.take(2):
#     plt.figure()
#     plt.imshow(image)
#     plt.title(get_label_name(label))
# plt.show()
######## Check several images ######## END


######## Preprocessing for images ######## START
IMG_SIZE = 160 # All images will be resized to 160x160
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
######## Preprocessing for images ######## END


######## Shuffle and assignment ######## START
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
for image_batch, label_batch in train_batches.take(1):
   pass
print(image_batch.shape)
######## Shuffle and assignment ######## END
