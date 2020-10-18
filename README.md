## Introduction

A repository contains WIP code for 2020 NAIC competitions. Competition details can be found at https://naic.pcl.ac.cn/frame/2.



## Install

### Requirements

Testing environment:

- CentOS Linux 7 (Core)
- Python 3.6
- PyTorch 1.6
- CUDA 10.1
- segmentation-model-pytorch 0.1.2

### Installation

a. Create a conda virtual environment and activate it.

```shell
conda create -n naic python=3.6 -y
conda activate naic
```

b. Install PyTorch and torchvision following the official instructions. Here we use PyTorch and CUDA 10.1. You may also switch to other vision by specifying the version number.

```shell
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

c. Install project requirements by.

```she
pip install -r requirements.txt
```

## Getting Started

### Prepare datasets

Project has following structure. Dataset is recommended to be placed at $PROJECTROOT/data although it can be configured via `cfg.ini`.

```shell
├── cfg.ini
├── inference.py
├── plot_latest_log.sh
├── requirements.txt
├── src
│   ├── data
│   │   ├── aug.py
│   │   ├── dataset.py
│   │   ├── histequal.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── logger.py
│   ├── loss.py
│   ├── metrics.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── utils.py
├── tools
│   ├── check_mask.py
│   ├── correct_small_area.py
│   ├── count_connected_component.py
│   ├── __init__.py
│   ├── mixup.py
│   ├── plot_logs.py
│   └── voting.py
├── data
│   ├── image_A
│   ├── results
│   └── train
│       ├── image
│       ├── label
│       └── train_aug
│           ├── image
│           └── label
└── train.py
```

### Training

We created `label.csv` file to split data into train and validation set and used this label file in our dataset class `src/data/dataset.py`.

```shell
image,mask,label
48366.tif,48366.png,train
60305.tif,60305.png,train
267981.tif,267981.png,train
230520.tif,230520.png,val
432758.tif,432758.png,val
49156.tif,49156.png,val
```

You can also write your own dataset class and ignore this file. Please change line 162 and line 172 correspondingly in `train.py` if you used your own dataset.

Currently, we are using segmentation-model-pytorch API for our model's architectures and encoders. All supportive models can be found at https://github.com/qubvel/segmentation_models.pytorch.

#### Train a model

You can set your training parameters via CML and training a model like:

```shell
python train.py --encoder resnext50_32x4d -w imagenet --arch unet -b 64 -lr 5e-5 -wd 5e-6 --num_workers 12 --num_epoch 100 --parallel
```

By default, model will be saved at `$PROJECTROOT/checkpoints` and log will be saved at `$PROJECTROOT/logs`. You can resume your model via `--reusme`.

#### Inference

We provide inference script to output the prediction to desired directory. Simple using the following command to set `checkpoint`, `input_dir` and `output_dir`.

```shell
python inference.py --checkpoint checkpoints/deeplabv3plus_efficientnet-b3/202010041423/epoch_209_0.8191.pth --input_dir data/image_A --output_dir ./results
```



## Acknowledgement

```
@misc{Yakubovskiy:2019,
  Author = {Pavel Yakubovskiy},
  Title = {Segmentation Models Pytorch},
  Year = {2020},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
}
```

 