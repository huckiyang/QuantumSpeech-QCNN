# Quantum Deep Learning for Speech 
Quantum Machine Learning for Automatic Spoken-Term Recognition. 

- **NEW** Our paper is accepted to IEEE International Conference on Acoustics, Speech, & Signal Processing ([ICASSP](https://scholar.google.com/citations?view_op=top_venues&hl=en&venue=HHC6AUo36fEJ.2020&vq=phy_acousticssound)) 2021.

<ins style="display:block;">
<noscript>
    <a style="position: relative; display: inline-block; cursor: pointer;" target="_blank" href="https://polo.feathr.co/v1/analytics/crumb?a_id=588f5ef88e80271ed938605b&cpn_id=60772f265738c36951564557&crv_id=6082ec83d696fb5e0c058e4b&flvr=ad_click&e_id=5f4ea53926d08e58c8a388ca&t_id=60772f285738c3695156455e&rdr=1&p_id=6081bf2b61ffff4537c5a6af">
        <img src="https://djhofpfq0ge2i.cloudfront.net/banners/e1hpatcz2/images/png?t=1619192972107" style="width:500px;height:100px;border:none;" width="500" height="100"/>
    </a>
    <img src="https://marco.feathr.co/v1/refresh" border=0 width=0 height=0 />
    <img src="https://polo.feathr.co/v1/analytics/crumb?a_id=588f5ef88e80271ed938605b&cpn_id=60772f265738c36951564557&crv_id=6082ec83d696fb5e0c058e4b&flvr=ad_view&e_id=5f4ea53926d08e58c8a388ca&t_id=60772f285738c3695156455e&p_id=6081bf2b61ffff4537c5a6af" border=0 width=0 height=0 />
</noscript>
</ins>

We would like to thank the reviewers and committee members in the Speech Processing and Quantum Signals community. 

Released the quantum speech processing code! (2020 Dec) [Colab demo](https://colab.research.google.com/drive/11Yi53W6-Z-uW84A8Sr6OqOMlBIel1PKM?usp=sharing) is also provided. ICASSP [Video](https://youtu.be/ZigIaFFFUhw) | [Slides](https://docs.google.com/presentation/d/1wHWnx1KXbzPe_YX-QXayISVBgLNa9IYe3EF3Jd42VZY/edit?usp=sharing)

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/images/demo.png" width="200">

- [Preprint Link](https://arxiv.org/abs/2010.13309) "Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition" 


## 1. Environment

<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" />

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
cd ../dataset
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

- run the ctc model in `./speech_quantum_dl`

```shell
python qsr_ctc_wer.py
```

### Result pre-trained weight in `checkpoints/asr_ctc_demo.hdf5`

```python
Epoch 32/50
107/107 [==============================] - 5s 49ms/step - loss: 0.1191 - val_loss: 0.7115
Epoch 33/50
107/107 [==============================] - 5s 49ms/step - loss: 0.1547 - val_loss: 0.6701
=== WER: 9.895833333333334  % 
```

Tutorial Link. 

- Only for academic purpose. Feel free to contact the author for the other purposes.

## Reference

If you think this work helps your research or use the code, please consider reference our paper. Thank you!

```bib
@inproceedings{yang2021decentralizing,
  title={Decentralizing feature extraction with quantum convolutional neural network for automatic speech recognition},
  author={Yang, Chao-Han Huck and Qi, Jun and Chen, Samuel Yen-Chi and Chen, Pin-Yu and Siniscalchi, Sabato Marco and Ma, Xiaoli and Lee, Chin-Hui},
  booktitle={2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6523--6527},
  year={2021},
  organization={IEEE}
}
```

## Federated Learning and Virtualization

We use [PySyft](https://github.com/OpenMined/PySyft) for vertical federated learning setup. Please refer to veritical learning [example](https://github.com/OpenMined/PySyft/tree/dev/examples/vertical-learning) for decentralized learning.

## Acknowledgment 

We would like to appreciate [Xanadu AI](https://www.xanadu.ai/) for providing the [PennyLane](https://pennylane.ai/) and [IBM research](https://www.research.ibm.com/) for providing [qiskit](https://qiskit.org/) and quantum hardware to the community. There is no conflict of interest.

## FAQ

Since the area between speech and quantum ML is still quite new, please feel free to open a [issue](https://github.com/huckiyang/speech_quantum_dl/issues) for discussion. 

Feel free to use this implementation for other speech processing or sequence modeling tasks (e.g., speaker recognition, speech seperation, event detection ...) as the quantum advantages discussed in the [paper](https://arxiv.org/abs/2010.13309).
