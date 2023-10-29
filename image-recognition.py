import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained MobileNet model for image recognition
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
model = hub.load(model_url)

# Function to perform image recognition
def recognize_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)

    predicted_class = model(image)
    _, class_idx = tf.math.top_k(predicted_class, k=1)
    
    return class_idx

# Load and recognize an image
image_path = "image.jpg"  # Replace with the path to your image
class_idx = recognize_image(image_path)

# Map class index to labels (you may need to adjust this based on the specific model used)
labels_path = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels = tf.keras.utils.get_file("ImageNetLabels.txt", labels_path)
with open(labels) as file:
    labels_list = file.read().splitlines()

# Display the recognized object
recognized_label = labels_list[class_idx[0][0]]
print(f"Recognized object: {recognized_label}")

# Display the image with the recognized object label
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image)
plt.imshow(image)
plt.title(recognized_label)
plt.show()
