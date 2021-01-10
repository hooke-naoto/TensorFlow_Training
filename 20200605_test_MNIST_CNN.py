######## Import ########
print('Import...')
import tensorflow as tf
import argparse
# "argparse" enables you to specify parameters when you run this program from command line (Windows: command Prompt, Mac Terminal).
# In this case, "pochs_num" and "batch_size_num" can be specified for each running.
# In case of no specifiation from command line, default values are 5 and 128 (you can refer below).
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs_num', help='Epoch number to fit the model.', required=False, type=int, default=5)
parser.add_argument('--batch_size_num', help='Batch size to fit the model.', required=False, type=int, default=128)
args = parser.parse_args()
epochs_num = args.epochs_num
batch_size_num = args.batch_size_num
######## Import ######## END

######## Load data ########
print('\nLoad data...')
# All images and labels from MNIST are NumPy n-dimentional arrays.
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# All images shall be standardized as 28x28 with 0 to 1 (from 8bit: 0 to 255).
# print('\n[Before Reshape]')
# print('type(train_images):\n', type(train_images))
# print('type(test_images):\n', type(test_images))
# print('train_images.shape:\n', train_images.shape)
# print('test_images.shape:\n', test_images.shape)
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
# print('\n[After Reshape]')
# print('type(train_images):\n', type(train_images))
# print('type(test_images):\n', type(test_images))
# print('train_images.shape:\n', train_images.shape)
# print('test_images.shape:\n', test_images.shape)
# Categorize train/test labels.
# print('\n[Before to_categorical]')
# print('type(train_labels):\n', type(train_labels))
# print('type(test_labels):\n', type(test_labels))
# print('train_labels.shape:\n', train_labels.shape)
# print('test_labels.shape:\n', test_labels.shape)
# print('train_labels:\n', train_labels)
# print('test_labels:\n', test_labels)
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
# print('\n[After to_categorical]')
# print('type(train_labels):\n', type(train_labels))
# print('type(test_labels):\n', type(test_labels))
# print('train_labels.shape:\n', train_labels.shape)
# print('test_labels.shape:\n', test_labels.shape)
# print('train_labels:\n', train_labels)
# print('test_labels:\n', test_labels)
# print('\n')
######## Load data ######## END

######## NN definition ########
print('\nNN definition...')
model = tf.keras.models.Sequential([
    # (28, 28, 1)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),    # Conv2D: Depth 32, Patch-size 3x3, Stride 1, No Padding
    # (26, 26, 32)
    tf.keras.layers.MaxPooling2D((2, 2)),
    # (13, 13, 32)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),    # Conv2D: Depth 32, Patch-size 3x3, Stride 1, No Padding
    # (11, 11, 64)
    tf.keras.layers.MaxPooling2D((2, 2)),
    # (5, 5, 64)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # (3, 3, 64)
    tf.keras.layers.Flatten(),
    # (576)
    tf.keras.layers.Dense(64, activation='relu'),
    # (64)
    tf.keras.layers.Dense(10, activation='softmax'),
    # (10)
    ])
model.summary()
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
######## NN definition ######## END

######## NN training ########
print('\nNN training...')
model.fit(train_images, train_labels, epochs=epochs_num, batch_size=batch_size_num)
######## NN training ######## END

######## NN evaluation ######## END
print('\nNN evaluation...')
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
######## NN evaluation ######## END
