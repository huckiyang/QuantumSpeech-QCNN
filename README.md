# Speech Quantum Deep Learning
Quantum Machine Learning for Speech Processing.

- Official Code will be released on December 27th 2020. 

<img src="https://github.com/huckiyang/speech_quantum_dl/blob/main/demo.png" width="200">

- [Paper Link](https://arxiv.org/abs/2010.13309) "Decentralizing Feature Extraction with Quantum Convolutional Neural Network for Automatic Speech Recognition" 

## Environment

- option 1: from conda and pip install
```bash
conda install -c anaconda tensorflow-gpu=2.0
conda install -c conda-forge scikit-learn 
conda install -c conda-forge librosa 
pip install pennylane --upgrade 
```

- option 2: from environment.yml (for 2080 Ti with CUDA 10.0) 
```bash
conda env create -f environment.yml
```

Origin with tensorflow 2.0 with CUDA 10.0.

## Dataset

Google [Speech Commands Dataset V1](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

```shell
mkdir ../dataset
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar -xf speech_commands_v0.01.tar.gz
```

### Audio Features Extration

### Quanvolution

## Training

### QCNN-RNN Attention Model

- Use Additional U-Net

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

