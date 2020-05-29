import tensorflow as tf
mnist = tf.keras.datasets.mnist

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs_num', help='Epoch number to fit the model.', required=False, type=int, default=5)
parser.add_argument('--batch_size_num', help='Batch size to fit the model.', required=False, type=int, default=128)
args = parser.parse_args()
epochs_num = args.epochs_num
batch_size_num = args.batch_size_num

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('-------- model.fit --------')
model.fit(train_images, train_labels, epochs=epochs_num, batch_size=batch_size_num)
print('-------- model.evaluate --------')
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Show images
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
