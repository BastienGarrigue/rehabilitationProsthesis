import keras
import numpy as np
import cv2
import re
from skimage import io
import os



def gaussian_blur(img):
    image = np.array(img)
    image_blur = cv2.GaussianBlur(image,(65,65),0.01)
    new_image = image_blur
    return new_image


def dataset_augmentation(DIR_DATASET, IMG_SIZE):
    # i = number of existing samples in each class
    i = 10
    for files in os.listdir(DIR_DATASET):
        image = DIR_DATASET + "/" + files
        image = keras.preprocessing.image.load_img(image, target_size=(IMG_SIZE, IMG_SIZE))
        files_array = keras.preprocessing.image.img_to_array(image)
        augmented_image = gaussian_blur(files_array)
        r = re.compile("([a-zA-Z]+)([0-9]+)")
        word_label = r.match(files).groups()[0]
        io.imsave(DIR_DATASET + "/" + word_label + "" + str(i) + '.png', augmented_image)
