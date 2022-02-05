# Advanced Machine Learning Project - Domain-To-Text 

### Assignment submission by:
### - 291018
### - 287639
### - 282870

# General Info

## PACS path

Make sure PACS folder is in this directory

## Pretrained models

In order to reproduce the values reported in the table, you have to download the pretrained models from this link: https://drive.google.com/drive/folders/17tWDDDPY9fRLrnL3YbwkHrilq12oii2M?usp=sharing

Then, you have to put the "outputs" folder in this directory

## Torch "bug"

For some reason torch==1.4.0 is too old to pull pretrained models from pytorch hub, so we used torch==1.9.0 (which comes with torchvision==0.10.0 and has CUDA 11.1 support for RTX). The rest of the requirements (specified in *_requirements.txt_*) are fine.

(We also used tensorboard so make sure to have it installed via *_pip install tensorboard_*)

## Data labels
Labels containing visual descriptions of the appearance and the style of each image are in *_'./datalabels/_* folder (each text file is a label, the path to each text file mimics the path of the corresponding image in the PACS folder). Each label describes in order:
```
Level of details
Edges
Color saturation
Color shades
Background
Single instance
Text
Texture
Perspective
```

## Finetuned model weights with the new dataset
Create a folder *_'./metric_learning/'_*, download from the link below the file *_'549_BEST_checkpoint.pth'_* and put it inside the created folder.

Download link: https://drive.google.com/file/d/1oXyCr_bkqzcdoJhbTWn6srUNHe2FXt9s/view?usp=sharing

# Original README

Basic code to reproduce the baselines (point 1 of the project). 

## Dataset

1 - Download PACS dataset from the portal of the course in the "project_topics" folder.

2 - Place the dataset in the DomainToText_AMLProject folder making sure that the images are organized in this way:

```
PACS/kfold/art_painting/dog/pic_001.jpg
PACS/kfold/art_painting/dog/pic_002.jpg
PACS/kfold/art_painting/dog/pic_003.jpg
...
```

## Pretrained models

In order to reproduce the values reported in the table, you have to download the pretrained models from this link: https://drive.google.com/drive/folders/17tWDDDPY9fRLrnL3YbwkHrilq12oii2M?usp=sharing

Then, you have to put the "outputs" folder into 

```
/DomainToText_AMLProject/
```


## Environment

To run the code you have to install all the required libraries listed in the "requirements.txt" file.

For example, if you read

```
torch==1.4.0
```

you have to execute the command:

```
pip install torch==1.4.0
```

