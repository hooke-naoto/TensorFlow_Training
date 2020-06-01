######## Preparation ########
print('Preparation...')
from tensorflow.keras.datasets import reuters
import numpy as np
import tensorflow as tf
######## Preparation ######## END

######## Load data ########
print('Load data...')
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print('train_data:', str(len(train_data)))    # 8982
print('test_data:', str(len(test_data)))    # 2246
######## Load data ######## END

######## Data convert ########
print('Data convert to string...')
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join(
                            [reverse_word_index.get(i - 3, '?') for i in train_data[0]]
                            )
# print(decoded_newswire)
######## Data convert ######## END

######## Data vecrorization ########
print('Data vectorization...')
# Convert the rawa data (list of words index) to vector data.
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.    # Set the index of results[i] as 1.
    return results
# Vectorize train data and test data.
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# print(x_train[0].shape)    # (25000, 10000)

# There are 2 types for vectorization of labels

# Vectorize train_labels and test_labels as "one-hot encoding".
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
#### manual description ####
# def one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results
# y_train = one_hot(train_labels)
# y_test = one_hot(test_labels)
#### manual description #### END

# Vectorize train_labels and test_labels as integer vector.
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
######## Data vecrorization ######## END

######## NN definiion ########
print('NN definition...')
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
  tf.keras.layers.Dense(64, activation='relu'),    # Try "4" to check the bottleneck behavior.
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(46, activation='softmax')
  ])
# Model compile
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss=tf.keras.losses.categorical_crossentropy,    # For one-hot encoding
    # loss=tf.keras.losses.sparse_categorical_crossentropy,    # For integer vector
    metrics=['accuracy'])
######## NN definiion ######## END

######## Study for approach ########
print('Study for approach...')
x_train_validation = x_train[:1000]
y_train_validation = y_train[:1000]
x_train_partial = x_train[1000:]
y_train_partial = y_train[1000:]
history = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=8,    # 20 at "Study of approach" -> 8 at "Implementation"
    batch_size=512,
    validation_data=(x_train_validation, y_train_validation)
    )
results = model.evaluate(x_test, y_test)
print(results)
import matplotlib.pyplot as plt
loss = history.history['loss']
loss_validation = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, loss_validation, 'b', label='Validation loss')
plt.title('Training loss & Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
acc_validation = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, acc_validation, 'b', label='Validation accuracy')
plt.title('Training accuracy & Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
######## Study for approach ######## END

######## Evaluation ########
predictions = model.predict(x_test)
print(predictions[0])    # A vector with 46 items.
print('The index of max value:', str(np.argmax(predictions[0])))
######## Evaluation ######## END
