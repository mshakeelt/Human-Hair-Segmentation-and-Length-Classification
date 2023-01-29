from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
from skimage import io
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

width,height = 256, 256
dim = (width, height)

def segmentation_inference(image_path, model_path):
    input_image = io.imread(image_path)
    input_image = cv2.resize(input_image, dim, interpolation=cv2.INTER_AREA)
    input_image = scaler.fit_transform(input_image.reshape(-1, input_image.shape[-1])).reshape(input_image.shape)
    input_image = preprocess_input(input_image)  #Preprocess based on the pretrained backbone...
    input_image = np.expand_dims(input_image, axis=0)
    
    model = load_model(model_path, compile=False)
    prediction = model.predict(input_image)
    prediction = np.argmax(prediction, axis=3)

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.title('Testing Image')
    plt.imshow(input_image[0])
    plt.subplot(122)
    plt.title('Testing Label')
    plt.imshow(prediction[0])
    plt.show()