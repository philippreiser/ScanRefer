# SparseScanRefer: Visual Grounding in RGB-D Scans with SparseConv and Dual-Set Clustering


## Introduction

We  fuse  a  new  detection  module  into  ScanRefer by substituting the current PointNet++ and VoteNet based architecture with the novel Instance Segmentation ap-proach  of  PointGroup  which  demonstrated  new  SOTA results on ScanNet v2 and S3DIS (3D Instance Segmentation).

## Setup
```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

Install the necessary packages for ScanRefer listed out in `requirements.txt`:
```shell
pip install -r requirements.txt
```
Afterwards follow the PG instructions

__Before moving on to the next step, please don't forget to set the project root path to the `CONF.PATH.BASE` in `lib/config.py`.__

### Data preparation
1. Download the ScanRefer dataset and unzip it under `data/`. 
2. Download the preprocessed [GLoVE embeddings (~990MB)](http://kaldir.vc.in.tum.de/glove.p) and put them under `data/`.
3. Download the ScanNetV2 dataset and put (or link) `scans/` under (or to) `data/scannet/scans/` (Please follow the [ScanNet Instructions](data/scannet/README.md) for downloading the ScanNet dataset).
> After this step, there should be folders containing the ScanNet scene data under the `data/scannet/scans/` with names like `scene0000_00`
4. Pre-process ScanNet data. A folder named `scannet_data/` will be generated under `data/scannet/` after running the following command. Roughly 3.8GB free space is needed for this step:
```shell
cd data/scannet/
python batch_load_scannet_data.py
```
> After this step, you can check if the processed scene data is valid by running:
> ```shell
> python visualize.py --scene_id scene0000_00
> ```

## Usage
### Training
To train the SparseScanRefer model with RGB values:
```shell
python scripts/script1.py
```
For more training options (batch_size, fix_pg,..), please run `scripts/train.py -h`.

For additional detail, please see the ScanRefer and PointGroup papers:
"[ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/abs/1912.08830)"  
"[PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation](https://arxiv.org/abs/2004.01658)"

Copyright (c) 2020 Dave Zhenyu Chen, Angel X. Chang, Matthias Nießner
