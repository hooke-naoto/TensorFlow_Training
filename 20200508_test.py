######## Import ########
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import time
######## Import ######## END


######## Download classifier ########
# Ref: https://tfhub.dev/s?module-type=image-classification&q=tf2
# e.g. [TF2] Imagenet (ILSVRC-2012-CLS) classification with MobileNet V2.
classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
# e.g. Imagenet (ILSVRC-2012-CLS) classification with Inception ResNet V2.
classifier_url ="https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/4" #@param {type:"string"}
print("classifier_url:")
print(classifier_url)
######## Download classifier ######## END


######## Define the NN model ########
IMAGE_SHAPE = (224, 224)
nn_model = tf.keras.Sequential([
                                hub.KerasLayer(classifier_url,
                                input_shape=IMAGE_SHAPE+(3,))
                                ])
######## Define the NN model ######## END


######## Try an image ########
# Import modules.
import numpy as np
import PIL.Image as Image # PIL = Python Image Library
# Prepare an image.
Hektor = Image.open('Hektor.jpg').resize(IMAGE_SHAPE) # Resize as 224x224.
Hektor = np.array(Hektor)/255.0 # Scale as 0 to 1.
# Predict a highest confidence class with the classifier.
print("Hektor[np.newaxis, ...]:")
print(Hektor[np.newaxis, ...])
result = nn_model.predict(Hektor[np.newaxis, ...]) # "result" contains confidences for all 1000 classes.
predicted_class = np.argmax(result[0], axis=-1)
with open("result.txt", mode='w') as f:
    f.write("New file")
# # Check the labels.
# labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# imagenet_labels = np.array(open(labels_path).read().splitlines())
imagenet_labels = np.array(open("ImageNetLabels.txt").read().splitlines())
print("imagenet_labels:")
print(imagenet_labels)
# Show the result.
plt.imshow(Hektor)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title()) # Underscore "_" means the output value won't be used to save memory.
plt.show()
######## Try an image ######## END


######## Export the NN model ########
import time
t = time.time()
export_path = "tmp/saved_models/{}".format(int(t))
nn_model.save(export_path, save_format='tf')
######## Export the NN model ######## END


######## Reload and predict then compare with prebious results ########
nn_model_reloaded = tf.keras.models.load_model(export_path)
reloaded_result_batch = nn_model_reloaded.predict(Hektor[np.newaxis, ...])
result_batch = nn_model.predict(Hektor[np.newaxis, ...])
print("Deviation between original and reloaded:")
print(abs(reloaded_result_batch - result_batch).max()) # Must be zero (no deviation between reloaded model and saved model.)
######## Reload and predict then compare with prebious results ######## END
