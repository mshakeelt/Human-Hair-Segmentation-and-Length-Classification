import os
from unicodedata import name
import numpy as np

from matplotlib import pyplot as plt
import segmentation_models as sm
sm.set_framework('tf.keras')
import random
import splitfolders

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import MeanIoU

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


seed=24
batch_size= 16
n_classes=3

#Use this to preprocess input for transfer learning
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

def train_val_split():
    input_folder = 'segmentation_dataset\\'
    output_folder = 'segmentation_dataset\\data_for_training_and_testing\\'
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.90, .10), group_prefix=None) # default values


def preprocess_data(img, mask, num_class):
    #Scale images
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
    #Convert mask to one-hot
    mask = to_categorical(mask, num_class)

    return (img,mask)

def trainGenerator(train_img_path, train_mask_path, num_class):

    img_data_gen_args = dict(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range = 0.1,
                             fill_mode='reflect')

    image_datagen = ImageDataGenerator(**img_data_gen_args)
    mask_datagen = ImageDataGenerator(**img_data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        train_img_path,
        class_mode = None,
        batch_size = batch_size,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        batch_size = batch_size,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)

    for (img, mask) in train_generator:
        img, mask = preprocess_data(img, mask, num_class)
        yield (img, mask)


def segmentation_training(train_img_gen, val_img_gen):
    x, y = train_img_gen.__next__()
    num_train_imgs = len(os.listdir('segmentation_dataset\\data_for_training_and_testing\\train_images\\train\\'))
    num_val_images = len(os.listdir('segmentation_dataset\\data_for_training_and_testing\\val_images\\val\\'))
    steps_per_epoch = num_train_imgs//batch_size
    val_steps_per_epoch = num_val_images//batch_size

    IMG_HEIGHT = x.shape[1]
    IMG_WIDTH  = x.shape[2]
    IMG_CHANNELS = x.shape[3]
    print(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                    classes=n_classes, activation='softmax')
    model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

    print(model.summary())
    print(model.input_shape)

    history=model.fit(train_img_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=10,
            verbose=1,
            validation_data=val_img_gen,
            validation_steps=val_steps_per_epoch)

    model.save('face_hair_segmentation_10_epochs_UNET_RESNET_backbone_batch16.hdf5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']

    plt.plot(epochs, acc, 'y', label='Training IoU')
    plt.plot(epochs, val_acc, 'r', label='Validation IoU')
    plt.title('Training and validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.show() 


def semantic_evaluation(val_img_gen):
    model = load_model("face_hair_segmentation_10_epochs_UNET_RESNET_backbone_batch16.hdf5", compile=False)
    test_image_batch, test_mask_batch = val_img_gen.__next__()
    test_mask_batch_argmax = np.argmax(test_mask_batch, axis=3)
    test_pred_batch = model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=3)
    IOU_keras = MeanIoU(num_classes=n_classes)
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())
    img_num = random.randint(0, test_image_batch.shape[0]-1)
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_image_batch[img_num])
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask_batch_argmax[img_num])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_pred_batch_argmax[img_num])
    plt.show() 
    
    
    
if __name__ == '__main__':
    train_img_path = "segmentation_dataset\\data_for_training_and_testing\\train_images\\"
    train_mask_path = "segmentation_dataset\\data_for_training_and_testing\\train_masks\\"
    train_img_gen = trainGenerator(train_img_path, train_mask_path, num_class=3) 

    val_img_path = "segmentation_dataset\\data_for_training_and_testing\\val_images\\"
    val_mask_path = "segmentation_dataset\\data_for_training_and_testing\\val_masks\\"
    val_img_gen = trainGenerator(val_img_path, val_mask_path, num_class=3)

