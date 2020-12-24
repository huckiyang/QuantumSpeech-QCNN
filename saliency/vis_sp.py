import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from cam_sp import layer_output
from tensorflow import keras

labels = [
    'left', 'go', 'yes', 'down', 'up', 'on', 'right', 'no', 'off', 'stop',
]

SAVE_PATH = "data_quantum/" # Data saving folder
q_train = np.load(SAVE_PATH + "q_train_speech_all.npy")
x_train = np.load(SAVE_PATH + "x_train_speech_all.npy")
qm_train = np.load(SAVE_PATH + "pub_bogota_train_speech.npy")
y_train = np.load(SAVE_PATH + "y_train_speech_all.npy")
qkr3_train = np.load(SAVE_PATH + "q_train_speech_all_kr3.npy")
model = keras.models.load_model('checkpoints/1014_1231_base_sp2cmd.hdf5') # 1014_1231
#for i in range(10):
idx = 1
cmd_n = labels[np.argmax(y_train[idx], axis=0)]
print("cmd name = ",cmd_n) ## audio is "yes"

plt.figure()
plt.subplot(2, 2, 1)
a = 12 # front size
plt.imshow(librosa.power_to_db(x_train[idx,:,:,0], ref=np.max))
plt.xticks([])
plt.yticks([])

plt.title('(a) Input Mel-Spectrogram', fontsize=a)
_, conv_out = layer_output(x_train, model, "batch_normalization_6",k=idx)
plt.subplot(2, 2, 2)
plt.imshow(librosa.power_to_db(conv_out[0,:,:,0], ref=np.max))
plt.xticks([])
plt.yticks([])

plt.title('(b) 2x2 Neural-Conv Encoded', fontsize=a)

plt.subplot(2, 2, 3)
plt.imshow(librosa.power_to_db(q_train[idx,:,:,0], ref=np.max))
plt.xticks([])
plt.yticks([])

plt.title('(c) 2x2 Quantum-Conv Encoded', fontsize=a)
#plt.subplot(5, 1, 4)
#librosa.display.specshow(librosa.power_to_db(qm_train[idx,:,:,0], ref=np.max))
#plt.title('2x2 Quantum Conv Encoded (IBMQ-noisy)', fontsize=20)
plt.subplot(2, 2, 4)
plt.imshow(librosa.power_to_db(qkr3_train[idx,:,:,0], ref=np.max))
plt.xticks([])
plt.yticks([])
plt.title('(d) 3x3 Quantum-Conv Encoded', fontsize=a)

plt.tight_layout()
plt.savefig("images/icassp_0_a_ok"+ cmd_n + ".pdf")
