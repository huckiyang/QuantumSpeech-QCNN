from tensorflow import keras
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
import os
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import string
from models import build_asr_model
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"

labels = ['go',  'up', 'on', 'no']
characters = string.ascii_lowercase # set(char for label in labels for char in label)

# Mapping characters to integers
char_to_num = L.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = L.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


b_size = 64
MAX_word_length = 2 # for demo using a fixed length. May modify it with lambda function in CTC_layer.
SAVE_PATH = "data_quantum/asr_set/"
load_asr_data = True

if load_asr_data == True:
    print("Load pre-processed speech data for CTC loss WER test")
    new_x_tr = np.load(SAVE_PATH + "asr_x_tr.npy")
    new_x_val = np.load(SAVE_PATH + "asr_x_val.npy")
    new_q_tr = np.load(SAVE_PATH + "asr_q_tr.npy")
    new_q_val = np.load(SAVE_PATH + "asr_q_val.npy")
    with open(SAVE_PATH + "char_y_val.json", 'r') as f:
        char_y_val = json.load(f)
    with open(SAVE_PATH + "char_y_tr.json", 'r') as f:
        char_y_tr = json.load(f)

else:
    print("Please process your own features.")
    exit()

print("-- Validation Size: ", np.array(char_y_val).shape, np.array(new_x_val).shape, np.array(new_q_val).shape)
print("-- Training Size: ", np.array(char_y_tr).shape, np.array(new_x_tr).shape, np.array(new_q_tr).shape)

# Get the QCNN-ASR-CTC model
model = build_asr_model(30, 63, 4) # 60 126 1
model.summary()


def encode_single_sample(img, label):
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # Return a dict as our model is expecting two inputs
    return {"speech": img, "label": label}

print("=== Making CTC input dataset ...")

train_dataset = tf.data.Dataset.from_tensor_slices((new_q_tr, char_y_tr))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(b_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((new_q_val, char_y_val))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(b_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

epochs = 50
early_stopping_patience = 25
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

history = model.fit(train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],)


# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="speech").input, model.get_layer(name="dense2").output
)

prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred, max_length=MAX_word_length):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

val_port = 3
pred_texts = []

for batch in validation_dataset.take(val_port):
    batch_speech = batch["speech"]
    preds = prediction_model.predict(batch_speech)
    pred_texts.append(decode_batch_predictions(preds))

import itertools
cor_idx = 0
pred_texts = list(itertools.chain.from_iterable(pred_texts))

for idx, word in enumerate(char_y_val[0:b_size*val_port]):
    if word != pred_texts[idx]:
        cor_idx += 1

print(pred_texts)
print("=== QCNN-ASR WER:", 100*cor_idx/len(pred_texts), " %")
model.save('checkpoints/' + 'ctc_asr_demo.hdf5')
