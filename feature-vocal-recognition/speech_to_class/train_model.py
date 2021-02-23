from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from rehabilitationProsthesis.speech_to_class.load_dataset import load_dataset
import numpy as np


def train_model(DIR_DATASET, IMG_SIZE):
    dataset = load_dataset(DIR_DATASET, IMG_SIZE)
    trainImages = np.array([i[0] for i in dataset]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    trainLabels = np.array([i[1] for i in dataset])

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation = 'softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(trainImages, trainLabels, batch_size = 16, epochs = 10, verbose = 1,validation_split=0.25,)
    return model.save("model.h5")

