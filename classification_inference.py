from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io


model = load_model(r"trained_models\best_classifier_0.76.hdf5", compile=False)


image_path = r"classification_dataset\long\long_01916.bmp"
input_image = io.imread(image_path)

input_image = np.expand_dims(input_image, axis=0)
prediction = model.predict(input_image)
print(prediction)
prediction = np.argmax(prediction, axis=1)
print(prediction)