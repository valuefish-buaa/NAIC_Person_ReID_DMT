# NAIC_Person_ReID_Competition For 2020

**This repository contains our source code  for the 2020 Person ReID Compitition of NAIC. 
该代码在初赛A榜单排35， B榜单排26**

## Authors

- [Chen Baoying](https://github.com/beibuwandeluori)
- [Zheng Siting](https://github.com/minilionChang)
- [Song Yusheng](https://github.com/wolkenmeer911)

## Introduction

Detailed information about the 2020 Person ReID Compitition of NAIC can be found [here](https://www.datafountain.cn/competitions/454).

The code is modified from [NAIC_Person_ReID_DMT](https://github.com/heshuting555/NAIC_Person_ReID_DMT)

## Useful Tricks

- [x] DataAugmention(RandomErase + ColorJittering +RandomAffine + RandomHorizontallyFlip + Padding + RandomCrop)
- [x] WarmUp + MultiStepLR 
- [x] Ranger
- [x] ArcFace
- [x] Faster Reranking
- [x] Gem
- [x] Weighted Triplet Loss
- [x] Remove Long Tail Data (pid with single image)
- [x] Solving UDA through Generating Pseudo Label  
- [x] Distmat Ensemble
- [x] FP16
- [x] Backbones: resnet + efficientnet 

1. Due to the characteristics of the dataset, Luo find color Jittering can greatly improve model performance. 
2. Luo rewrote Faster Reranking, using the GPU to calculate the distance, and using sparse matrix storage, which can save GPU memory and RAM to meet the organizer's hardware requirements.
3. Pseudo label is a common trick used in deep learning competitions. We use a trained model to generate pseudo label and add some constraints to get the cleaner label.( we didn't use)
4. Luo use Ranger optimizer to make the model converge faster and better.
5. FP16 training can save 30% memory cost and  is 30% faster with no precision drop. Although we didn't have time to try it in this competition, it proved to be a very useful thing in other competitions. 
Recommend everyone to use. You can refer to [apex](https://github.com/NVIDIA/apex) install it. if you don't have apex installed, please turn-off FP16 training by setting SOLVER.FP16=False (We didn't use)
6. We use efficientnet as backbone
### Project File Structure

```
+-- NAIC_Challenge
|   +-- NAIC_Person_ReID_DMT(put code here)
|   +-- model(dir to save the output)
|   +-- data
|		+--train
|		+--test
|			+--query_a
|			+--gallery_a
|			+--query_b
|			+--gallery_b
```

## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone https://github.com/heshuting555/NAIC_Person_ReID_DMT.git`

3. Install dependencies:
   - [pytorch>=1.1.0](https://pytorch.org/)
   - python>=3.5
   - torchvision
   - [yacs](https://github.com/rbgirshick/yacs)
   - cv2
   - efficientnet_pytorch
   
   We use cuda 9.0/python 3.6.7/torch 1.2.0/torchvision 0.4.0 for training and testing.
   
5.  [ResNet-ibn](https://github.com/XingangPan/IBN-Net) is applied as the backbone. Download ImageNet pretrained model  [here](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) 

## RUN

1. If you want to get the same score as online in the Person ReID Compitition of NAIC . Use the following commands:

   ```bash
   bash run.sh
   ```

2. If  you want to use our baseline for training. 

   ```bash
   python train.py --config_file [CHOOSE WHICH config TO RUN]
   # E.g
   #python train.py --config_file configs/naic_round2_model_a.yml
   ```

3. If  you want to use our UDA method for training. 

   ```bash
   python train_UDA.py --config_file [CHOOSE WHICH config TO RUN] --config_file_test [CHOOSE WHICH CONFIG TO TEST and GET PSEUDO LABLE] --data_dir_query [PATH TO QUERY DATASET] --data_dir_gallery [PATH TO GALLERY DATASET]
   # E.g
   #python train_UDA.py --config_file configs/naic_round2_model_b.yml --config_file_test configs/naic_round2_model_a.yml --data_dir_query ../data/test/query_a --data_dir_gallery ../data/test/gallery_a
   ```

4. If  you want to test the model and get the result in json format required by the competition.

   ```bash
   python test.py --config_file [CHOOSE WHICH CONFIG TO TEST]
   # E.g
   #python test.py --config_file configs/naic_round2_model_a.yml
   ```

### Citation

If you find our work useful in your research, please consider citing:
```
@InProceedings{He_2020_CVPR_Workshops,
author = {He, Shuting and Luo, Hao and Chen, Weihua and Zhang, Miao and Zhang, Yuqi and Wang, Fan and Li, Hao and Jiang, Wei},
title = {Multi-Domain Learning and Identity Mining for Vehicle Re-Identification},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
@InProceedings{Luo_2019_CVPR_Workshops,
author = {Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
title = {Bag of Tricks and a Strong Baseline for Deep Person Re-Identification},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```
