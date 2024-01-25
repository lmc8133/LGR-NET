# LGR-NET
The source code of our paper: LGR-NET: Language Guided Reasoning Network for Referring Expression Comprehension (under review)
# Installation
## 1. Prepare the environment
```
python==3.7.13
torch==1.11.0+cu113
torchvision==0.12.0+cu113
mmcv-full==1.3.18
tensorboard==2.8.0
transformers==2.11.0
einops==0.4.1
icecream==2.1.2
numpy==1.22.3
scipy==1.8.0
ftfy==6.1.1
```
We recommand to install mmdet from the source code in this repository (./models/swin_model)

## 2. Dataset preparation
We follow the data preparation of TransVG, which can be found in [GETTING_STARTED.md](https://github.com/djiajunustc/TransVG/blob/main/docs/GETTING_STARTED.md)

## 3. Checkpoint preparation
We use the checkpoint from [QRNet](https://github.com/LukeForeverYoung/QRNet), they put the checkpoints in [Google Drive](https://drive.google.com/drive/folders/1GTi32iEfsJdYNtcHCUQIbhMdL5YFByVF). The downloaded checkpoints should be put in `checkpoint/`.

## Training
```
bash train.sh
```

## Evaluation
```
bash test.sh
```

## Acknowledge
This code is partially based on [TransVG](https://github.com/djiajunustc/TransVG) and [QRNet](https://github.com/LukeForeverYoung/QRNet)