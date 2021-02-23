from speech_to_class.extract_melscale import extract_melscale
from speech_to_class.dataset_augmentation import dataset_augmentation
from speech_to_class.train_model import train_model
import constants as c


def launch():
    extract_melscale(c.DIR_AUDIO, c.DIR_DATASET)
    dataset_augmentation(c.DIR_DATASET, c.IMG_SIZE)
    train_model(c.DIR_DATASET, c.IMG_SIZE)


if __name__ == '__main__':
    launch()

