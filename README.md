# Speech Quantum Deep Learning
Quantum Machine Learning for Speech Recognition.

- **NEW** Released the quantum speech processing code! (12/24) 

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/images/demo.png" width="200">

- [Paper Link](https://arxiv.org/abs/2010.13309) "Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition" 

## 1. Environment

- option 1: from conda and pip install
```python
conda install -c anaconda tensorflow-gpu=2.0
conda install -c conda-forge scikit-learn 
conda install -c conda-forge librosa 
pip install pennylane --upgrade 
```

- option 2: from environment.yml (for 2080 Ti with CUDA 10.0) 
```python
conda env create -f environment.yml
```

Origin with tensorflow 2.0 with CUDA 10.0.

## 2. Dataset

We use Google [Speech Commands Dataset V1](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) for Limited-Vocabulary Speech Recognition.

```shell
mkdir ../dataset
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar -xf speech_commands_v0.01.tar.gz
```

### 2.1. Pre-processed Features

We provide `2000` pre-processed feautres in `./data_quantum`, which included both mel features, and `(2,2)` quanvolution features with `1500` for training and `500` for testing. You could get `90.6%` test accuracy by the provided data.                              

You could use `np.load` to load these features to train your own quantum speech processing model in [`3.1`](https://github.com/huckiyang/speech_quantum_dl/blob/main/README.md#3-training). 

### 2.2. Audio Features Extraction (optional)

Please set the sampling rate `sr` and data ratio (`--port N` for 1/N data; `--port 1` for all data) for extracting Mel Features.

```python
python main_qsr.py --sr 16000 --port 100 --mel 1 --quanv 1
```

### 2.3. Quanvolution Encoding (optional)

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/images/Quanv.png" width="150">

If you have pre-load audio features from `2.2.` you can set the quantum convolution kernal size in `helper_q_tool.py` function [quanv](https://github.com/huckiyang/speech_quantum_dl/blob/main/helper_q_tool.py#L47). We provide an example for kernal size = 3 in [line 57](https://github.com/huckiyang/speech_quantum_dl/blob/main/helper_q_tool.py#L57).

You will see a message below during the Quanvolution Encoding with features extraction comment from `2.2.`.

```python
===== Shape 60 126
Kernal =  2
Quantum pre-processing of train Speech:
2/175
```

## 3. Training

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/images/QCNN_Sys_ASR.png" width="400">

### 3.1 QCNN U-Net Bi-LSTM Attention Model

Spoken Terms Recognition with additional [U-Net Encoder](https://arxiv.org/abs/2010.13309) discussed in our work.

```shell
python main_qsr.py
```

In 25 epochs. One way to improve the recognition system performance is to encode more data for training, refer to `2.2.` and `2.3`.

```python
1500/1500 [==============================] - 3s 2ms/sample - val_loss: 0.4408 - val_accuracy: 0.9060                              
```

- Alternatively, training without U-Net as the method proposed in [Douglas C. de Andrade et al.](https://arxiv.org/abs/1808.08929) similar to their [implementation](https://github.com/douglas125/SpeechCmdRecognition) but without `kapre` layers.

Please set `use_Unet = False.` in [model.py](https://github.com/huckiyang/speech_quantum_dl/blob/main/models.py#L81).

```python
def attrnn_Model(x_in, labels, ablation = False):
    # simple LSTM
    rnn_func = L.LSTM
    use_Unet = False
```
### 3.2 Neural Saliency by Class Activation Mapping (CAM)

```shell
python cam_sp.py
```

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/images/cam_sp_0.png" width="350">

### 3.3 CTC Model for Automatic Speech Recognition 

We also provide a CTC model with Word Error Rate (WER) evaluation for future studies to the community refer to the [discussion](https://arxiv.org/abs/2010.13309). 

For example, an output "y-e--a" of input "yes" is identified as an incorrect word with the CTC alignment.

Noted this Quantum ASR CTC version is only supported `tensorflow-gpu==2.3`. Please create a new environment for running this experiment.

- unzip the features for asr

```
cd data_quantum/asr_set
bash unzip.sh
```

- run the ctc model in `./`

```shell
python qsr_ctc_wer.py
```

### result pre-trained weight in `checkpoints/asr_ctc_demo.hdf5`

```python
Epoch 38/50
107/107 [==============================] - 5s 49ms/step - loss: 0.1097 - val_loss: 0.5990
=== WER: 11.97916  %
```

Tutorial Link. 

- Only for academic purpose. Feel free to contact the author for the other purposes.

## Reference

If you think this work helps your research or use the code, please consider reference our paper. Thank you!

```bib
@article{yang2020decentralizing,
  title={Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition},
  author={Yang, Chao-Han Huck and Qi, Jun and Chen, Samuel Yen-Chi and Chen, Pin-Yu and Siniscalchi, Sabato Marco and Ma, Xiaoli and Lee, Chin-Hui},
  journal={arXiv preprint arXiv:2010.13309},
  year={2020}
}
```

## Acknowledgment 

We would like to appreciate [Xanadu AI](https://www.xanadu.ai/) for providing the [PennyLane](https://pennylane.ai/) and [IBM research](https://www.research.ibm.com/) for providing [qiskit](https://qiskit.org/) and quantum hardware to the community. There is no conflict of interest.

## FAQ

Since the area between speech and quantum ML is still quite new, please feel free to open a [issue](https://github.com/huckiyang/speech_quantum_dl/issues) for discussion. 

Feel free to use this implementation for other speech processing or sequence modeling tasks (e.g., speaker recognition, speech seperation, event detection ...) as the quantum advantages discussed in the [paper](https://arxiv.org/abs/2010.13309).
