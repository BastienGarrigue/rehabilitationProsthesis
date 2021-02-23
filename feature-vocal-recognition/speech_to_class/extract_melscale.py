import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os


def extract_melscale(DIR_AUDIO, DIR_DATASET):
    i = 0
    hop_length = 512
    n_mels = 128
    n_fft = 2048
    for folder in os.listdir(DIR_AUDIO):
        for sample in os.listdir(DIR_AUDIO+"/"+folder):
            y, sr = librosa.load(DIR_AUDIO+"/"+folder+"/"+sample)
            command, _ = librosa.effects.trim(y)
            S = librosa.feature.melspectrogram(command, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length);
            plt.savefig(DIR_DATASET + '/' + folder + '' + str(i) + '.png')
            i = i + 1

