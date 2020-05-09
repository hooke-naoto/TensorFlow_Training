######## Import ######## START
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
######## Import ######## END


######## Download classifier ######## START
# e.g. mobilenet_v2
# Ref: https://tfhub.dev/s?module-type=image-classification&q=tf2
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
                                    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
                                    ])
######## Download classifier ######## END


######## Try an image "Grace Hopper" ######## START
# Import modules.
import numpy as np
import PIL.Image as Image # PIL = Python Image Library
# Prepare an image "Grace Hopper".
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE) # Resize as 224x224.
grace_hopper = np.array(grace_hopper)/255.0 # Scale as 0 to 1.
# Predict a highest confidence class with the classifier.
result = classifier.predict(grace_hopper[np.newaxis, ...]) # "result" contains confidences for all 1000 classes.
predicted_class = np.argmax(result[0], axis=-1)
# Check the labels.
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
print("labels_path:")
print(labels_path)
imagenet_labels = np.array(open(labels_path).read().splitlines())
print("imagenet_labels:")
print(imagenet_labels)
# Show the result.
# plt.imshow(grace_hopper)
# plt.axis('off')
# predicted_class_name = imagenet_labels[predicted_class]
# _ = plt.title("Prediction: " + predicted_class_name.title()) # Underscore "_" means the output value won't be used to save memory.
# plt.show()
######## Try an image "Grace Hopper" ######## END


######## Transfer Learning with flower images/labels ######## START
# Prepare flower images.
data_root = tf.keras.utils.get_file(
                                    'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                    untar=True)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255) # Scale
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE) # Resize
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

# Predict with not re-trained NN model.
result_batch = classifier.predict(image_batch)
predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)
# plt.figure(figsize=(10,9))
# plt.subplots_adjust(hspace=0.5)
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.imshow(image_batch[n])
#     plt.title(predicted_class_names[n])
#     plt.axis('off')
# _ = plt.suptitle("ImageNet predictions")
# plt.show()

# Prepare the headless NN model which just extracts features.
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

# Prepare the feature extractor layer as a not trainable layer.
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
feature_batch = feature_extractor_layer(image_batch)
feature_extractor_layer.trainable = False
print(feature_batch.shape) # Size of features for each image should be 1280.

# Add the feature extractor layer to the headless NN model.
model = tf.keras.Sequential([
                                feature_extractor_layer,
                                layers.Dense(image_data.num_classes, activation='softmax')
                                ])
model.summary()
predictions = model(image_batch)
predictions.shape

# Configure the revised NN model (not yet retrained).
model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss='categorical_crossentropy',
                metrics=['acc'])

# Retrain the revised NN model with callback.
class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])
        self.model.reset_metrics()
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)
batch_stats_callback = CollectBatchStats()
history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])

# Show history graphs for losses and accuracies.
# plt.figure()
# plt.ylabel("Loss")
# plt.xlabel("Training Steps")
# plt.ylim([0,2])
# plt.plot(batch_stats_callback.batch_losses)
# plt.show()
# plt.figure()
# plt.ylabel("Accuracy")
# plt.xlabel("Training Steps")
# plt.ylim([0,1])
# plt.plot(batch_stats_callback.batch_acc)
# plt.show()

# Check results.
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print("class_names:")
print(class_names)
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print("predicted_label_batch:")
print(predicted_label_batch)
label_id = np.argmax(label_batch, axis=-1)
print("label_id:")
print(label_id)
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = "green" if predicted_id[n] == label_id[n] else "red"
    plt.title(predicted_label_batch[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.show()

# Export the retrained NN model.
import time
t = time.time()
export_path = "/tmp/saved_models/{}".format(int(t))
model.save(export_path, save_format='tf')
print(export_path)

# Load and predict then compare with prebious results.
reloaded = tf.keras.models.load_model(export_path)
result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)
abs(reloaded_result_batch - result_batch).max() # Must be zero (no deviation between reloaded model and saved model.)

######## Transfer Learning with flower images/labels ######## END
