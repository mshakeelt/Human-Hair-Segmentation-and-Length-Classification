import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip
import splitfolders

def classification_augmentation(folder_reduced, folder_augmented):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant')
    allimages = os.listdir(folder_reduced)
    print('Total images', len(allimages))
    for image in allimages:
        img = load_img(os.path.join(folder_reduced, image)) # This is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 224, 224)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 224, 224)
        for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_augmented, save_prefix=os.path.splitext(image)[0], save_format='bmp'):
            break

def gray_mask_to_catagorical_converter(mask_path):
    msk_list = os.listdir(mask_path)
    for mask in msk_list:
        img_path = os.path.join(mask_path, mask)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rescalled_image = np.zeros(img.shape, dtype=int)
        rescalled_image[img >= 175] = 2
        rescalled_image[(img >= 75) & (img <= 174)] = 1
        rescalled_image[img <= 74] = 0
        cv2.imwrite(img_path, rescalled_image)

def load_data(path):
    gray_images = sorted(glob(os.path.join(path, "gray")+'\*.jpg'))
    rgb_images = sorted(glob(os.path.join(path, "rgb")+'\*.jpg'))
    return gray_images, rgb_images

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def segmentation_augmentation(images, masks, save_path, augment=True):
    H = 224
    W = 224

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("\\")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("\\")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x2, x3, x4, x5]
            save_masks =  [y, y2, y3, y4, y5]
        else:
            save_images = [x]
            save_masks = [y]


        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"

            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

            image_path = os.path.join(save_path, "rgb", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

def image_converter(gray_images, rgb_images):
    for gray, rgb in tqdm(zip(gray_images, rgb_images), total=len(gray_images)):
        gray_name = gray.split("\\")[-1].split(".")
        rgb_name = rgb.split("\\")[-1].split(".")
        gray_image_name = gray_name[0]
        rgb_image_name = rgb_name[0]
        gray_img = cv2.imread(gray)
        rgb_image = cv2.imread(rgb)
        tmp_gray_img_name = f"{gray_image_name}.png"
        tmp_rgb_img_name = f"{rgb_image_name}.png"
        gray_path = os.path.join('segmentation_dataset', "gray", tmp_gray_img_name)
        rgb_path = os.path.join('segmentation_dataset', "rgb", tmp_rgb_img_name)
        cv2.imwrite(gray_path, gray_img)
        cv2.imwrite(rgb_path, rgb_image)

def single_dir_converter(rgb_images):
    for rgb in rgb_images:
        rgb_name = rgb.split("\\")[-1].split(".")
        rgb_image_name = rgb_name[0]
        rgb_image = cv2.imread(rgb)
        tmp_rgb_img_name = f"{rgb_image_name}.png"
        rgb_path = os.path.join('segmentation_dataset', "images", tmp_rgb_img_name)
        cv2.imwrite(rgb_path, rgb_image)

def color_to_gray_mask_converter(folder_mask, destination, dim):
    allmasks = os.listdir(folder_mask)
    for mask in allmasks:
        image_bmp = mask.replace(".png", ".bmp")
        img = cv2.imread(os.path.join(folder_mask, mask))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        converted_image = np.zeros(img.shape, dtype=int)
        converted_image[img[:,:,2] >= 5] = (255,255,255)
        converted_image[img[:,:,1] >= 5] = (128,128,128)
        cv2.imwrite(os.path.join(destination, image_bmp), converted_image)

def rgb_to_gray_converter(rgb_folder_path, gray_folder_path):
    rgb_images = os.listdir(rgb_folder_path)
    for rgb_image in tqdm(rgb_images,  total=len(rgb_images)):
        image_path = os.path.join(rgb_folder_path, rgb_image)
        img_rgb = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(gray_folder_path, rgb_image), img_gray)

def rename_dataset(gray, rgb, masks):
    gray_files = os.listdir(gray)
    rgb_files = os.listdir(rgb)
    masks_files = os.listdir(masks)
    counter = 1
    for index, _ in enumerate(masks_files):
        if rgb_files[index].replace('.jpg', '.png') == masks_files[index]:
            oldgrayfile = os.path.join(gray, gray_files[index])
            oldrgbfile = os.path.join(rgb, rgb_files[index])
            oldmaskfile = os.path.join(masks, masks_files[index])
            newgrayfile = os.path.join(gray, f'long_{counter:05d}.png')
            newrgbfile = os.path.join(rgb, f'medium_{counter:05d}.jpg')  #
            newmaskfile = os.path.join(masks, f'medium_{counter:05d}.png')
            os.rename(oldgrayfile, newgrayfile)
            os.rename(oldrgbfile, newrgbfile)
            os.rename(oldmaskfile, newmaskfile)
            counter += 1

def image_resizer(source_dir, dest_dir, target_dim):
    original_files = os.listdir(source_dir)
    for original_file in original_files:
        image_png = original_file.replace(".jpg", ".png")
        image_path = os.path.join(source_dir, original_file)
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, target_dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dest_dir, image_png), resized_image)

def train_val_split():
    input_folder = 'classification_dataset\\'
    output_folder = 'classification_dataset\\data_for_training_and_testing\\'
    # Split with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.90, .10), group_prefix=None) # default values