import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pennylane import numpy as np
from scipy.io import wavfile
import warnings
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, SGD
## Local Definition 
from data_generator import gen_mel
from models import cnn_Model, rnn_Model, dense_Model, attrnn_Model
from helper_q_tool import gen_qspeech, plot_acc_loss, show_speech
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time as ti
data_ix = ti.strftime("%m%d_%H%M")

labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

train_audio_path = '../dataset/'
SAVE_PATH = "../quantum_speech/data_quantum/" # Data saving folder

sr = 16000 # sampling rate
model_use = 1 # use of model
port = 1
n_eps = 3 # number of epochs
b_size = 16 # batch size
compute_train = False
compute_quanv = False

def gen_train(labels, train_audio_path, sr, port):
    all_wave, all_label = gen_mel(labels, train_audio_path, sr, port)

    label_enconder = LabelEncoder()
    y = label_enconder.fit_transform(all_label)
    classes = list(label_enconder.classes_)
    y = keras.utils.to_categorical(y, num_classes=len(labels))

    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    h_feat, w_feat, _ = x_train[0].shape
    np.save(SAVE_PATH + "n_x_train_speech.npy", x_train)
    np.save(SAVE_PATH + "n_x_test_speech.npy", x_valid)
    np.save(SAVE_PATH + "n_y_train_speech.npy", y_train)
    np.save(SAVE_PATH + "n_y_test_speech.npy", y_valid)
    print("===== Shape", h_feat, w_feat)

    return x_train, x_valid, y_train, y_valid

def gen_quanv(x_train, x_valid, kr):
    print("Kernal = ", kr)
    q_train, q_valid = gen_qspeech(x_train, x_valid, kr)

    np.save(SAVE_PATH + "pub_bogota_train_speech.npy", q_train)
    # np.save(SAVE_PATH + "pub_rome_test_speech.npy", q_valid)

    return q_train, q_valid

if compute_train == True:
    x_train, x_valid, y_train, y_valid = gen_train(labels, train_audio_path, sr, port) 
else:
    x_train = np.load(SAVE_PATH + "x_train_speech_all.npy")
    x_valid = np.load(SAVE_PATH + "x_test_speech_all.npy")
    y_train = np.load(SAVE_PATH + "y_train_speech_all.npy")
    y_valid = np.load(SAVE_PATH + "y_test_speech_all.npy")


if compute_quanv:
    q_train, q_valid = gen_quanv(x_train, x_valid, 2) # or kernal = 3
else:
    q_train = np.load(SAVE_PATH + "q_train_speech_all.npy")
    q_valid = np.load(SAVE_PATH + "q_test_speech_all.npy")

## For Quanv Exp.
early_stop = EarlyStopping(monitor='val_loss', mode='min', 
                           verbose=1, patience=10, min_delta=0.0001)

checkpoint = ModelCheckpoint('checkpoints/final.hdf5', monitor='val_acc', 
                             verbose=1, save_best_only=True, mode='max')


if model_use == 0:
    model = dense_Model(x_train[0], labels)
elif model_use == 1:
    model = attrnn_Model(q_train[0], labels)

model.summary()

history = model.fit(
    x=q_train, 
    y=y_train,
    epochs=n_eps, 
    callbacks=[checkpoint], 
    batch_size=b_size, 
    validation_data=(q_valid,y_valid)
)


v_model.save('checkpoints/'+ data_ix + '_sr.hdf5')

print("=== Batch Size: ", b_size)