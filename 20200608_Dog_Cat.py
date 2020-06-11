######## Import ########
print('Import...')
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
######## Import ######## END

######## Data loading ########
print('Data loading...')
dir_data_original = '/Users/hooke/Downloads/dogs-vs-cats/train'
dir_data_small = '/Users/hooke/Downloads/dogs-vs-cats_small'
dir_data_original = r'C:\Users\naoto\Downloads\dogs-vs-cats\train'
dir_data_small = r'C:\Users\naoto\Downloads\dogs-vs-cats_small'
dir_train = os.path.join(dir_data_small, 'train')
dir_validation = os.path.join(dir_data_small, 'validation')
dir_test = os.path.join(dir_data_small, 'test')
dir_train_cats = os.path.join(dir_train, 'cats')
dir_train_dogs = os.path.join(dir_train, 'dogs')
dir_validation_cats = os.path.join(dir_validation, 'cats')
dir_validation_dogs = os.path.join(dir_validation, 'dogs')
dir_test_cats = os.path.join(dir_test, 'cats')
dir_test_dogs = os.path.join(dir_test, 'dogs')
if False == os.path.exists(dir_data_small):
    os.mkdir(dir_data_small)
    os.mkdir(dir_train)
    os.mkdir(dir_validation)
    os.mkdir(dir_test)
    os.mkdir(dir_train_cats)
    os.mkdir(dir_train_dogs)
    os.mkdir(dir_validation_cats)
    os.mkdir(dir_validation_dogs)
    os.mkdir(dir_test_cats)
    os.mkdir(dir_test_dogs)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_train_cats, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_validation_cats, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_test_cats, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_train_dogs, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_validation_dogs, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(dir_data_original, fname)
    dst = os.path.join(dir_test_dogs, fname)
    shutil.copyfile(src, dst)
print('train cat:', len(os.listdir(dir_train_cats)))
print('train dog:', len(os.listdir(dir_train_dogs)))
print('validation cat:', len(os.listdir(dir_validation_cats)))
print('validation dog:', len(os.listdir(dir_validation_dogs)))
print('test cat:', len(os.listdir(dir_test_cats)))
print('test dog:', len(os.listdir(dir_test_dogs)))
######## Loading ######## END

######## NN definition ########
print('NN definition...')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
# model.summary()
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
    )
######## NN definition ######## END

######## Data preprocessing ########
print('Data preprocessing...')
data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
data_generator_validation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
    )
generator_train = data_generator_train.flow_from_directory(
    dir_train,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'    # Binary label for binary_crossentropy
    )
generator_validation = data_generator_validation.flow_from_directory(
    dir_validation,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'    # Binary label for binary_crossentropy
    )
# for data_batch, labels_batch in generator_train:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break
######## Data preprocessing ######## END

######## Data augmentation ########
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    )
######## Data augmentation ######## END

######## NN training ########
print('NN training...')
history = model.fit_generator(
    generator_train,
    steps_per_epoch=100,    # Step number for 2000 files with 20 batch files.
    epochs=30,
    validation_data=generator_validation,
    validation_steps=50
    )
model.save('dogs_and_cats_small_1.h5')    # Save as h5-file.
# print(history.history)    # Check keys of "history". (e.g. acc? accuracy?)
acc = history.history['acc']
acc_val = history.history['val_acc']
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='Validation accuracy')
plt.title('Training accuracy & Validation accuracy')
plt.legend()
plt.show()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training loss & Validation loss')
plt.legend()
plt.show()
######## NN training ######## END
