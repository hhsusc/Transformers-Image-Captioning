# Transforming Image Captioning: Integrating SwinV2, CSwin, and DeiT Architectures into the Pure Transformer (PureT) Model
Forked from the Implementation of __End-to-End Transformer Based Model for Image Captioning__ [[PDF/AAAI]](https://ojs.aaai.org/index.php/AAAI/article/view/20160) [[PDF/Arxiv]](https://arxiv.org/abs/2203.15350) [AAAI 2022] from [here](https://github.com/232525/PureT).\
Authors: Austin Lamb & Hassan Shah\
University of Southern California (USC)

![architecture](./imgs/architecture.png)

## Requirements (Our Main Enviroment)
+ Python 3.7.16
+ PyTorch 1.13.1
+ TorchVision 0.14.1
+ [coco-caption](https://github.com/tylin/coco-caption)
+ numpy 1.21.6
+ tqdm 4.66.1
+ transformers 4.30.2

See [env.yaml](env.yaml), [env.txt](env.txt), and [anaconda_env.yaml](anaconda_env.yaml) for more info.

## Preparation
### 1. coco-caption preparation
Refer coco-caption [README.md](./coco_caption/README.md), you will first need to download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models for use by SPICE. To do this, run:
```bash
cd coco_caption
bash get_stanford_models.sh
```
### 2. Data preparation
The necessary files in training and evaluation are saved in __`mscoco`__ folder, which is organized as follows:
```
mscoco/
|--feature/
    |--coco2014/
       |--train2014/
       |--val2014/
       |--test2014/
       |--annotations/
|--misc/
|--sent/
|--txt/
```
where the `mscoco/feature/coco2014` folder contains the raw image and annotation files of [MSCOCO 2014](https://cocodataset.org/#download) dataset. You can download a zip file from [here](https://drive.google.com/file/d/1b9fMeRiezu8oWD3eQOmBskEsDNHpT6pI/view?usp=sharing) and unzip at the root level of this repo.

__NOTE:__ You can also extract image features of MSCOCO 2014 using [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) or others and save them as `***.npz` files into `mscoco/feature` for training speed up, refer to [coco_dataset.py](datasets/coco_dataset.py) and [data_loader.py](datasets/data_loader.py) for how to read and prepare features. 
__In this case, you need to make some modifications to [pure_transformer.py](models/pure_transformer.py) (delete the backbone module). For you smart and excellent people, I think it is an easy work.__

### 3. Backbone Models Pre-trained Weights
Download pre-trained Backbone models from [here](https://drive.google.com/drive/folders/1ctV-DSFD_d_PxtgXyRAjouxqYRUmQO8z?usp=sharing) and place them in the root directory of this repo.

### 4. Pre-trained Image Captioning Models
You can download the saved models for each experiment [here](https://drive.google.com/drive/folders/1bMOdzRTYKITbJ-cRRtjuyVfycOx7Z3ZX?usp=sharing) and place within your experiments_PureT folder.

Note: If any of the links here don't work, you should find all the files in a folder at [this](https://drive.google.com/drive/folders/1azl9aC_s7lmY-r9TAKI4Gutng4hopFbm?usp=sharing) link.


## Training
*Note: our repository is mainly based on [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning), and we directly reused their config.yml files, so there are many useless parameter in our model. （__waiting for further sorting__）*

### 1. Training under XE loss
Before training, you can check and modify the parameters in `config.yml` and `train.sh` files. Then run the script(s) for each experiment:

```
# for XE training
bash experiments_PureT/PureT_XE/train.sh
bash experiments_PureT/PureT_SwinV2_XE/train.sh
bash experiments_PureT/PureT_CSwin_XE/train.sh
bash experiments_PureT/PureT_DeiT_XE/train.sh
```
### 2. Training using SCST (self-critical sequence training)
Copy the pre-trained model you saved from performing XE trainig into folder of `experiments_PureT/PureT_*_SCST/snapshot/` and modify `config.yml` and `train.sh` files to resume from the snapshot. If you download the already pre-trained model weights, then run the script (should already be setup to train from the downloaded pre-trained models):

```bash
# for SCST training
bash experiments_PureT/PureT_SCST/train.sh
bash experiments_PureT/PureT_SwinV2_SCST/train.sh
bash experiments_PureT/PureT_CSwin_SCST/train.sh
bash experiments_PureT/PureT_DeiT_SCST/train.sh
```

## Evaluation
Once you are done training (or if you downloaded the pre-trained models from [here](https://drive.google.com/drive/folders/1bMOdzRTYKITbJ-cRRtjuyVfycOx7Z3ZX?usp=sharing)) you can run inference/evaluation folowing this format: 

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py --folder experiments_PureT/PureT_SCST/ --resume 27
```

|BLEU-1|BLEU-2|BLEU-3|BLEU-4|METEOR|ROUGE-L| CIDEr |SPICE |
| ---: | ---: | ---: | ---: | ---: | ---:  | ---:  | ---: |
| 82.1 | 67.3 | 52.0 | 40.9 | 30.2 | 60.1  | 138.2 | 24.2 |


## Reference
Our work is forked and built off this resarch paper:
```
@inproceedings{wangyiyu2022PureT,
  title={End-to-End Transformer Based Model for Image Captioning},
  author={Yiyu Wang and Jungang Xu and Yingfei Sun},
  booktitle={AAAI},
  year={2022}
}
```

## Acknowledgements
This repository is based on [JDAI-CV/image-captioning](https://github.com/JDAI-CV/image-captioning), [ruotianluo/self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer), and [microsoft/CSWin-Transformer](https://github.com/microsoft/CSWin-Transformer).
