import numpy as np
from matplotlib import pyplot as plt
from skimage import io
import colorific
import colorsys
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import cv2
from matplotlib import pyplot as plt
import segmentation_models as sm
sm.set_framework('tf.keras')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)


def mask_to_categorical(mask):
    categorical = np.zeros(mask.shape, dtype=np.uint8)
    categorical[mask >= 175] = 2
    categorical[(mask >= 75) & (mask <= 174)] = 1
    categorical[mask <= 74] = 0
    return categorical

def categorical_to_mask(categorical):
    mask = np.zeros(categorical.shape, dtype=np.uint8)
    mask[categorical == 0] = 0
    mask[categorical == 1] = 128
    mask[categorical == 2] = 255
    return mask

def get_hair_pixels(rgb, mask):
    rgb[mask != 255] = 0
    return rgb

def norm_color(c):
    r, g, b = c
    return r / 255.0, g / 255.0, b / 255.0

def rgb_to_hex(color):
    return '#%.02x%.02x%.02x' % color

def draw_palette_as_image(palette):
    "Save palette as a PNG with labeled, colored blocks"
    size = (80 * len(palette.colors), 80)
    im = Image.new('RGB', size)
    draw = ImageDraw.Draw(im)
    for i, c in enumerate(palette.colors):
        v = colorsys.rgb_to_hsv(*norm_color(c.value))[2]
        (x1, y1) = (i * 80, 0)
        (x2, y2) = ((i + 1) * 80 - 1, 79)
        draw.rectangle([(x1, y1), (x2, y2)], fill=c.value)
        if v < 0.6:
            # white with shadow
            draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (90, 90, 90))
            draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value))
        else:
            # dark with bright "shadow"
            draw.text((x1 + 4, y1 + 4), rgb_to_hex(c.value), (230, 230, 230))
            draw.text((x1 + 3, y1 + 3), rgb_to_hex(c.value), (0, 0, 0))
    return im

def get_hair_color(hair_segment_array, min_color_distance, max_number_colors, zone):
    hair_segment_image= Image.fromarray(hair_segment_array)
    palette = colorific.palette.extract_colors(hair_segment_image, min_distance=min_color_distance, max_colors=max_number_colors)
    print("Detected Colors in zone", zone+1)
    for color in palette.colors:
        hex_color = colorific.palette.rgb_to_hex(color.value)
        percentage = round(color.prominence * 100, 2)
        print("Color: {} | Percentage: {}".format(hex_color, percentage))
    
    palette_image = draw_palette_as_image(palette)
    return palette_image


def segmentation_inference(input_image, model_path):
    input_image = cv2.resize(input_image, (256, 256), interpolation=cv2.INTER_AREA)
    input_image = scaler.fit_transform(input_image.reshape(-1, input_image.shape[-1])).reshape(input_image.shape)
    input_image = preprocess_input(input_image)  #Preprocess based on the pretrained backbone...
    input_image = np.expand_dims(input_image, axis=0)
    
    model = load_model(model_path, compile=False)
    mask = model.predict(input_image)
    mask = np.argmax(mask, axis=3)
    return mask[0]

def classification_inference(input_image, model_path):
    model = load_model(model_path, compile=False)
    input_image = cv2.resize(input_image, (224, 224), interpolation=cv2.INTER_AREA)
    input_image = np.expand_dims(input_image, axis=0)
    prediction = model.predict(input_image)
    prediction = np.argmax(prediction, axis=1)
    return prediction

def mask_divider(mask, pred_class):
    white_pixels = np.array(np.where(mask == (255, 255, 255) ))
    first_white_pixel = white_pixels[:,0]
    last_white_pixel = white_pixels[:,-1]
    first_white_pixel_row_index = first_white_pixel[0]
    last_white_pixel_row_index = last_white_pixel[0]
    pixels_to_divide = abs(first_white_pixel_row_index-last_white_pixel_row_index)
    
    divided_masks = []
    starting_index = first_white_pixel_row_index
    if pred_class == 0:
        print("Long hair detected!")
        print("Dividing hair pixels to 3 zones for color analysis!")
        per_image_rows = pixels_to_divide//3
        for _ in range(3):
            part_mask = np.zeros(mask.shape, dtype=np.uint8)
            part_mask[starting_index:starting_index+per_image_rows] = mask[starting_index:starting_index+per_image_rows]
            divided_masks.append(part_mask)
            starting_index += per_image_rows
    
    elif pred_class == 1:
        print("Medium hair detected!")
        print("Dividing hair pixels to 2 zones for color analysis!")
        per_image_rows = pixels_to_divide//2
        for _ in range(2):
            part_mask = np.zeros(mask.shape, dtype=np.uint8)
            part_mask[starting_index:starting_index+per_image_rows] = mask[starting_index:starting_index+per_image_rows]
            divided_masks.append(part_mask)
            starting_index += per_image_rows
    
    else:
        print("Short hair detected!")
        print("Moving on to color analysis!")
        divided_masks.append(mask)
    
    return divided_masks


def draw_visualization(visualizations): 
    plt.figure(figsize=(12, 8))
    nrows=len(visualizations)/2
    ncols=2
    titles = ['Input Image', "Mask", "Hair Segment", "Palette", "Hair Segment", "Palette", "Hair Segment", "Palette"]
    for index, image in enumerate(visualizations):
        plt.subplot(nrows, ncols, index+1)
        plt.title(titles[index])
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    rgb_image_path = r"test_images\IMG-20210924-WA0013.jpg"
    segmentation_model_path = r"trained_models\best_segmentor_0.86.hdf5"
    classification_model_path = r"trained_models\best_classifier_0.76.hdf5"

    rgb_image = io.imread(rgb_image_path)

    categorical_mask = segmentation_inference(rgb_image, segmentation_model_path) # Segmentation will create a catagorical mask in which pixel value 2 correspods to hair, 1 to face, and 0 to background
    mask_1Ch = categorical_to_mask(categorical_mask) # catagorical mask is converted to 0-255 range

    mask_3Ch = np.zeros((mask_1Ch.shape[0], mask_1Ch.shape[1], 3), dtype=np.uint8) # converting gray to 3 cheenel image as effecient net expect rgb image for classification
    mask_3Ch[:,:,0] = mask_1Ch
    mask_3Ch[:,:,1] = mask_1Ch
    mask_3Ch[:,:,2] = mask_1Ch

    predicted_class = classification_inference(mask_3Ch, classification_model_path) # classification will predict 1 of 3 classes. 0 for long, 1 for medium, 2 for short.
    
    mask = cv2.resize(mask_3Ch, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_AREA) # mask is reshaped to the input image shape
    divided_masks_per_classification = mask_divider(mask, predicted_class) # mask is divided into zones as per the recommendation of classifier
    
    
    visualizations_to_draw = []
    visualizations_to_draw.append(rgb_image)
    visualizations_to_draw.append(mask)    
    
    min_color_distance = 15 # Threshold to consider two colors distinct
    max_number_colors = 10  # Maximum number of colors to output in the palette
    
    for zone, part_mask in enumerate(divided_masks_per_classification):
        hair_segment = get_hair_pixels(rgb_image.copy(), part_mask)
        palette_img = get_hair_color(hair_segment, min_color_distance, max_number_colors, zone)
        visualizations_to_draw.append(hair_segment)
        visualizations_to_draw.append(palette_img)

    draw_visualization(visualizations_to_draw)
