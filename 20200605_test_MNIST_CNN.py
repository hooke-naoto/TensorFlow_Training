######## Import ########
print('Import...')
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs_num', help='Epoch number to fit the model.', required=False, type=int, default=5)
parser.add_argument('--batch_size_num', help='Batch size to fit the model.', required=False, type=int, default=128)
args = parser.parse_args()
epochs_num = args.epochs_num
batch_size_num = args.batch_size_num
######## Import ######## END

######## Load data ########
print('Load data...')
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
######## Load data ######## END

######## NN definition ########
print('NN definition...')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])
model.summary()
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
######## NN definition ######## END

######## NN training ########
print('NN training...')
model.fit(train_images, train_labels, epochs=5, batch_size=64)
######## NN training ######## END

######## NN evaluation ######## END
print('NN evaluation...')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
######## NN evaluation ######## END
