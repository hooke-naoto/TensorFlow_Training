######## Desctiption ########
# Goal: The detection camera for my cat in our house with RasPi as a stand-alone system (no web loading).
# Last: 20200508_ObjectDetection_simple.py ... Got customized "object detection" based on TensorFlow Hub examples.
# This: 20200509_ObjectDetection_local.py ... Remove web loading processes.
######## Desctiption ######## END

######## Parameters ########
save_result_image = True
show_result_image = True
min_score = 0.2 # Min scores for result image.
max_boxes = 10 # Max number of boxes for result image.
nn_model_path = "nn_model/ssd_mobilenet_v2_1"
nn_model_url = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
image_path = "Hektor.jpg"
######## Parameters ######## END

######## Setup ########
# General
import os
import time
import datetime
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# PILLOW (image processing)
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFont

# TensorFlow
import tensorflow as tf
import tensorflow_hub as tf_hub
######## Setup ######## END

######## Environment ########
print("\n[Environment]")
if tf.test.gpu_device_name() != "":
    print("GPU: %s" % tf.test.gpu_device_name())
else:
    print("GPU: none")
print("TensorFlow version:", tf.__version__)
######## Environment ######## END

######## Helper Functions ########
#### Load & Resize images ####
def load_and_resize_image(path, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix=".jpg")
    pil_image = Image.open(path)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(filename, format="JPEG", quality=90)
    print("Image loaded to %s." % filename)
    return filename
#### Load & Resize images #### END

#### Draw boxes ####
# Overlay labeled boxes on an image with formatted scores and label names.
def draw_boxes(image, boxes, class_names, scores, min_score, max_boxes):
    font = ImageFont.load_default()
    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, font, display_str_list=[display_str])
        np.copyto(image, np.array(image_pil))
    return image
#### Draw boxes #### END

#### Draw bounding box on image ####
# Adds a bounding box to an image.
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, font, thickness=2, display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill="lightgray")
    # If the total height of the display strings added to the top of the bounding box
    # exceeds the top of the image, stack the strings below the bounding box instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                       [(left, text_bottom - text_height - margin*2), (left + text_width, text_bottom)],
                       fill="lightgray"
                       )
        draw.text(
                  (left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font
                  )
        text_bottom -= text_height - margin*2
#### Draw bounding box on image #### END
######## Helper Functions ######## END

# Time record
t0 = time.time()

######## NN Model ########
print("\n[NN model loading]")
# Load NN model
# [ssd+mobilenet V2] was choosen because smaller and faster than [FasterRCNN+InceptionResNet V2].
try:
    nn_model = tf_hub.load(nn_model_path).signatures['default']
    print("NN model was loaded from local PC:", nn_model_path)
except OSError as error:
    print("NN model couldn't be loaded from local PC:", error)
    print("NN model will be loaded from TensorFlow Hub...")
    try:
        nn_model = tf_hub.load(nn_model_url).signatures['default']
        print("NN model was loaded from TensorFlow Hub:", nn_model_url)
    except OSError as error:
        print("Error - NN model loading:", error)
######## NN Model ######## END

# Time record
t1 = time.time()

######## Detection ########
print("\n[Detection]")
# ID based on date time
now = datetime.datetime.now()
ID = now.strftime("%Y%m%d_%H%M%S")
print("ID:", ID)

# Detect images in local PC and show results.
image_path_detection = load_and_resize_image(image_path, 400, 300)

# Time record
t2 = time.time()

# Run NN model.
image = tf.io.read_file(image_path_detection)
image = tf.image.decode_jpeg(image, channels=3)
image_converted  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
results = nn_model(image_converted)
results = {key:value.numpy() for key,value in results.items()}

# Show results.
print("%d objects were detected." % len(results["detection_scores"]))
######## Detection ######## END

# Time record
t3 = time.time()

######## Processing Time ########
print("\n[Processing Time]")
print("{:.2f}".format(t3 - t0), " sec")
print("NN model loading:", "{:.2f}".format(t1 - t0), "sec")
print("Image loading   :", "{:.2f}".format(t2 - t1), "sec")
print("Detection       :", "{:.2f}".format(t3 - t2), "sec")
######## Processing Time ######## END

######## Show & Save result image ########
image_with_boxes = draw_boxes(
                              image.numpy(),
                              results["detection_boxes"],
                              results["detection_class_entities"],
                              results["detection_scores"],
                              min_score,
                              max_boxes
                              )
fig = plt.figure(figsize=(8, 6))
plt.grid(False)
plt.imshow(image_with_boxes)
if save_result_image == True:
    folder_path = "DetectedImage/"
    if os.path.exists(folder_path) == False:
        os.mkdir(folder_path)
    plt.savefig(folder_path + ID + ".png")
if show_result_image == True:
    plt.show()
######## Show & Save result image ######## END
