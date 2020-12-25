# Speech Quantum Deep Learning
Quantum Machine Learning for Speech Processing.

- Official Code will be released on December 27th 2020. 

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

We provide 2000 pre-processed feautres in `./data_quantum`, which included both mel features, and `(2,2)` quanvolution features with `1500` for training and `500` for testing. You could get `85%` test accuracy these data.                              

You could use `np.load` to load these features to train your own quantum speech processing model in this repo. 

### 2.2. Audio Features Extration

### 2.3. Quanvolution

## 3. Training

### 3.1 QCNN-RNN Attention Model

Spoken Terms Recognition with additional [U-Net Encoder](https://arxiv.org/abs/2010.13309)

```shell
python main_qsr.py
```

In 30 epochs. One way to improve the recognition system performance is to encode more data for training, refer to `2.2.` and `2.3`.

```python
1500/1500 [==============================] - 3s 2ms/sample - val_loss: 0.7331 - val_accuracy: 0.8500                              
```

- without Additional U-Net as the baseline proposed in

Please set `use_Unet = False.` in [model.py](https://github.com/huckiyang/speech_quantum_dl/blob/main/models.py#L81).

```
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

We also provide a CTC model with Word Error Rate Evaluation for Quantum Speech Recognition. 

Tutorial Link. 

- Only for academic purpose. 

The author is affiliated with Georgia Tech.

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

