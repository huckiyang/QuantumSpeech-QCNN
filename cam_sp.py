from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models
import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

os.environ["CUDA_VISIBLE_DEVICES"]="0"
q_model = keras.models.load_model('checkpoints/0910_1843_qaunv_sp2cmd.hdf5') 
q_train = np.load("data_quantum/q_train_demo.npy")
x_train = np.load("data_quantum/x_train_demo.npy")
idx = 0 #  for grounded transcription command as "on"

def layer_output(in_feats, model, ly_name = "batch_normalization_6 ", k= idx):
    conv_layer = model.get_layer(ly_name)
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(in_feats[k:k+1])
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    return heatmap, conv_output

def vis_map(heatmap):
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    return heatmap

w_x, h_x = x_train[idx,:,:,0].shape

def to_rgb(heatmap):
    heatmap = np.uint8(255 * vis_map(np.rot90(heatmap[0])))
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.rot90(np.transpose(heatmap))]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((  h_x, w_x))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Save the superimposed image
    save_path = "images/color_cam.jpg"
    superimposed_img = keras.preprocessing.image.array_to_img(jet_heatmap)
    superimposed_img.save(save_path)

    cam_img= mpimg.imread(save_path)

    return cam_img

q_heatmap, _ = layer_output(q_train, q_model, "conv2d_2")
q_cam = to_rgb(q_heatmap)

x_model = keras.models.load_model('checkpoints/0910_1843_conv_sp2cmd.hdf5')
x_heatmap, _ = layer_output(x_train, x_model, "conv2d_7")
x_cam = to_rgb(x_heatmap)

c_model = keras.models.load_model('checkpoints/0910_1843_base_sp2cmd.hdf5')
c_heatmap, _ = layer_output(x_train, c_model, "conv2d_12")
c_cam = to_rgb(c_heatmap)

a = 12
plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(librosa.power_to_db(x_train[idx,:,:,0], ref=np.max))
plt.xticks([])
plt.yticks([])
plt.title('(a) Input Mel-Spectrogram', fontsize=a)
# plt.matshow(np.transpose(vis_map(x_train[idx,:,:,0])))
plt.subplot(2, 2, 2)
plt.imshow(q_cam)
plt.xticks([])
plt.yticks([])
plt.title('(b) Quanv + RNN (UAtt)', fontsize=a)
plt.subplot(2 ,2, 3)
plt.imshow(x_cam)
plt.xticks([])
plt.yticks([])
plt.title('(c) Conv + RNN (UAtt)', fontsize=a)
plt.subplot(2 ,2, 4)
plt.imshow(c_cam)
plt.xticks([])
plt.yticks([])
plt.title('(d) Baseline RNN (UAtt)', fontsize=a)
plt.tight_layout()
plt.savefig("images/cam_sp_"+str(idx)+".png")
