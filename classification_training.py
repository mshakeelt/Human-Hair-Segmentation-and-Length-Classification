from keras.applications.efficientnet import * #Efficient Net included here
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from sklearn import model_selection
from tqdm import tqdm
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt


input_shape = (224, 224, 3)
NUMBER_OF_CLASSES = 3

def build_model():
    conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gap"))
    #avoid overfitting
    model.add(layers.Dropout(rate=0.2, name="dropout_out"))
    # Set NUMBER_OF_CLASSES to the number of your final predictions.
    model.add(layers.Dense(NUMBER_OF_CLASSES, activation="softmax", name="fc_out"))
    conv_base.trainable = False

    print(model.summary())
    print(model.input_shape)
    model.compile(loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    metrics=["acc"])
    return model

def build_generators():
    TRAIN_IMAGES_PATH = r"classification_dataset\data_for_training_and_testing\train"
    VAL_IMAGES_PATH = r"classification_dataset\data_for_training_and_testing\val"


    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="constant",
    )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        TRAIN_IMAGES_PATH,
        # All images will be resized to target height and width.
        target_size=(224, 224),
        batch_size=16,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode="categorical",
    )

    validation_generator = test_datagen.flow_from_directory(
        VAL_IMAGES_PATH,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
    )
    return (train_generator, validation_generator)


def classification_training(train_generator, validation_generator, model):
    num_train_imgs = 7042*3
    num_val_images = 783*3
    steps_per_epoch = num_train_imgs//16
    val_steps_per_epoch = num_val_images//16

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=val_steps_per_epoch,
        verbose=1,
    )

    model.save('trained_models\\classification_10_epochs_effecientnetB0_224x224_backbone_batch16.hdf5')

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

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


   
if __name__ == '__main__':
    model = build_model()
    train_generator, validation_generator = build_generators()
    classification_training(train_generator, validation_generator, model)
