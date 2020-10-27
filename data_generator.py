import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import warnings
import tensorflow as tf


train_audio_path = '../dataset/'
sr=16000


def gen_mel(labels, train_audio_path, sr, port):
    all_wave = []
    all_label = []
    for label in tqdm(labels):
        waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
        for num, wav in enumerate(waves, 0):
            y, _ = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = sr)
            if num % port ==0:   # take 1/port samples
                if(len(y)== sr) :
                    mfcc_feat = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=128, power=1.0, n_mels=60, fmin=40.0, fmax=sr/2)
                    all_wave.append(np.expand_dims(mfcc_feat, axis=2))
                    all_label.append(label)
    
    return all_wave, all_label
