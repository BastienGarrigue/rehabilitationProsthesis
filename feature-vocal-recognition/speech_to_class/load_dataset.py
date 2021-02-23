from PIL import Image
import numpy as np
import os
import re


def label_img(name):

    r = re.compile("([a-zA-Z]+)([0-9]+)")
    word_label = r.match(name).groups()[0]
    if word_label == 'gauche':
        return np.array([1, 0])
    elif word_label == 'droite':
        return np.array([0, 1])


def load_dataset(DIR_DATASET, IMG_SIZE):
    dataset = []
    for img in os.listdir(DIR_DATASET):
        label = label_img(img)
        path = os.path.join(DIR_DATASET, img)
        if "DS_Store" not in path:
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
            dataset.append([np.array(img), label])
            return dataset
