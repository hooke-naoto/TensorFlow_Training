######## Import ########
print('Import...')
import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
######## Import ######## END

######## Data loading ########
print('Data loading...')
# Load dataset
## num_words ... Specify frequent words, remove other words.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
######## Data loading ######## END

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
# Vectorize train_labels and test_labels.
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# print(y_train)
######## Data vecrorization ######## END

######## NN creation ########
print('NN creation...')
# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
# Model compile
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# Model_reg
model_reg = tf.keras.models.Sequential([
    # tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='relu', input_shape=(10000,)),
    # tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l1(0.001), activation='relu'),
    # tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu', input_shape=(10000,)),
    # tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
    tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
# Model_reg compile
model_reg.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])
######## NN creation ######## END

######## NN trial ########
print('NN trial...')
# Separate 10000 data for validation.
x_train_val = x_train[:10000]
x_train_partial = x_train[10000:]
y_train_val = y_train[:10000]
y_train_partial = y_train[10000:]
# Training
model_fitted = model.fit(
    x_train_partial,
    y_train_partial,
    epochs=20,
    batch_size=512,
    validation_data=(x_train_val, y_train_val)
    )
model_fitted_reg = model_reg.fit(
    x_train_partial,
    y_train_partial,
    epochs=20,
    batch_size=512,
    validation_data=(x_train_val, y_train_val)
    )
# Export results
history_dict = model_fitted.history
acc = history_dict['accuracy']
acc_val = history_dict['val_accuracy']
loss_values = history_dict['loss']
loss_values_val = history_dict['val_loss']
#
history_dict_reg = model_fitted_reg.history
acc_reg = history_dict_reg['accuracy']
acc_val_reg = history_dict_reg['val_accuracy']
loss_values_reg = history_dict_reg['loss']
loss_values_val_reg = history_dict_reg['val_loss']
# Graph
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, acc_val, 'b', label='Validation acc')    # 'b': solid blue line
plt.plot(epochs, acc_reg, 'ro', label='Training acc (regularization)')
plt.plot(epochs, acc_val_reg, 'r', label='Validation acc (regularization)')
plt.title('Training accuracy & Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss_values, 'bo', label='Training loss')    # 'bo': blue dot
plt.plot(epochs, loss_values_val, 'b', label='Validation loss')    # 'b': solid blue line
plt.plot(epochs, loss_values_reg, 'ro', label='Training loss (regularization)')
plt.plot(epochs, loss_values_val_reg, 'r', label='Validation loss (regularization)')
plt.title('Training loss & Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
######## NN trial ######## END

######## NN training ########
# print('NN training...')
# # Training
# model.fit(x_train, y_train, epochs=4, batch_size=512)
######## NN training ######## END

######## NN results ########
# print('NN results...')
# # Results
# model.evaluate(x_test, y_test)
# # print(model.predict(x_test))
######## NN results ######## END

print('END!')
