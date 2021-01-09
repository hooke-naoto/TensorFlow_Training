######## Import ########
print('Import...')
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
######## Import ######## END

######## Data loading ########
print('Data loading...')
# For MacBook Pro
# dir_data_original = '/Users/hooke/Downloads/dogs-vs-cats/train'
# dir_data_small = '/Users/hooke/Downloads/dogs-vs-cats_small'
# For Windows Desktop PC
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
# Outputs will be as below.
# Input           150, 150, 3
# conv2d_1        148, 148, 32
# maxpooling2d_1  74, 74, 64
# conv2d_2        72, 72, 64
# maxpooling2d_2  36, 36, 128
# conv2d_3        34, 34, 64
# maxpooling2d_3  17, 17, 128
# conv2d_4        15, 15, 128
# maxpooling2d_4  7, 7, 128
# flatten_1       6272 (= 7 x 7 x 128)
# dense_1         512
# dense_2         1

model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
    )
######## NN definition ######## END

######## Data preprocessing ########
print('Data preprocessing...')

data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,    # Scaling as real number (NOT integer number).
    rotation_range=40,    #
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

generator_train = data_generator_train.flow_from_directory(
    dir_train,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'    # Binary label for binary_crossentropy
    )

data_generator_validation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
    )

generator_validation = data_generator_validation.flow_from_directory(
    dir_validation,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'    # Binary label for binary_crossentropy
    )

for data_batch, labels_batch in generator_train:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
######## Data preprocessing ######## END

######## Data augmentation ########
print('Data augmentation...')

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    )
######## Data augmentation ######## END

######## NN training ########
print('NN training...')

history = model.fit(
    generator_train,
    steps_per_epoch=100,    # Step number for 2000 files with 20 batch files.
    epochs=100,
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
