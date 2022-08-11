# Localization and direction of vessels with semantic segmentation by Unet
[![logo](https://img.shields.io/badge/HUANGYming-projects-orange?style=flat&logo=github)](https://github.com/HUANGYming) 

![](https://img.shields.io/badge/Linux%20build-pass-green.svg?logo=linux) 
![](https://img.shields.io/badge/NVIDIA-CUDA-green.svg?logo=nvidia) 

![](https://img.shields.io/badge/Python-3.6.13-green.svg?style=social&logo=python) 
![](https://img.shields.io/badge/anaconda-4.12.0-green.svg?style=social&logo=anaconda) 
![](https://img.shields.io/badge/Opencv-4.1.2.30-green.svg?style=social&logo=opencv) 
![](https://img.shields.io/badge/Pytorch-1.10.2-green.svg?style=social&logo=pytorch)
![](https://img.shields.io/badge/NumPy-1.19.2-green.svg?style=social&logo=NumPy)
![](https://img.shields.io/badge/TorchIO-1.3.5-green.svg?style=social&logo=torchio)
![](https://img.shields.io/badge/Pillow-8.2.0-green.svg?style=social&logo=torchio)



## Table of Contents
- [Result](#result)
- [Installation](#installation)
- [Structure](#structure)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)



## I. Result

| Class | IoU |
| ---- | ---- |
|Background|0.970|
|Vertical|0.757|
|Horizontal|0.786|
|||




### 1. Vertical vessel

![class2](https://yiminghku.oss-cn-hangzhou.aliyuncs.com/vertticalVessel.gif)

### 2. Horizontal vessel

![Class1](https://github.com/HUANGYming/Unet_multiclass/blob/main/actions/horizontalVessel.gif)

## II. Installation

**Python >= 3.6** ,Recommend to use Anaconda 
```
matplotlib==3.2.2
numpy==1.19.2
Pillow==8.2.0
pytorch-gpu==1.10.2
torchvision==0.4.2
tensorboard==2.6.0
future==0.18.2
tqdm==4.59.0
scikit-image==0.17.2
torchio==0.18.76
```



To install for Ubuntu,
```
$ conda install -r requirements.txt
```




## III. Structure
```
unet-multiclass-pytorch/
    - checkpoints/
    - data/
    - model/
    - runs/
    - Unet/
    - utils/
    - video/
    - params.json
    - README.md
    - requirements.txt
    - train.py
    - video_pre.py
```
in which:
- `checkpoints/` store the best models when training
- `data/` contains training data and masks
- `model/` contains the trained model
- `runs/` contains Tensorboard summary files
- `Unet/` contains U-Net structure
- `utils/` contains model parts and model related functions
- `video/` contains the video that video_pre.py needs
- `parameters.json` define all the parameters of the training and prediction
- `README.md` contains Tensorboard summary files
- `requirements.txt` contains the necessary packages
- `train.py` is the main script for model training
- `video_pre.py` is the main script for video prediction

## IV. Usage
### Example
| Parameter         |       Value      |
| ----              |       ----       |
|epoch              |       20         |
|batch_size         |       2          |
|learning_rate      |       0.1        |
|folder             | ./data/2classify |
|||

1. Terminal
```
python train.py -e 20 -b 2 -l 0.1 -folder ./data/2classify
```
2. JSON file (recommend)

| Mode              |       Explanation                  |
| ----              |       ----                         |
|train              |       training parameters           |
|prediction         |       prediction parameters         |
|change_label       |       change colors of labels      |
|||

![](https://yiminghku.oss-cn-hangzhou.aliyuncs.com/params.png)





## V. Reference

[1] vessel_net-main(Mingcong)

[2] https://github.com/France1/unet-multiclass-pytorch



## VI. License

HAUNGYIMING


