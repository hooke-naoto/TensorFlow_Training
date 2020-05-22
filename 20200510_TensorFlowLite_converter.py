import tensorflow as tf

saved_model_dir = "/Users/hooke/Documents/Coding/TensorFlow_Training/nn_model/ssd_mobilenet_v2_1/assets"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
