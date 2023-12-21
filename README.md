# DWGF
Code for the paper titled [A Diffusion Weighted Graph Framework for New Intent Discovery](https://arxiv.org/abs/2310.15836)

## Contents
[1. Data](#data)

[2. Requirements](#requirements)

[3. Running](#running)

[4. Thanks](#thanks)

[5. Citation](#citation)

## Data
We performed experiments on three public datasets: [clinc](https://aclanthology.org/D19-1131/), [banking](https://aclanthology.org/2020.nlp4convai-1.5/) and [stackoverflow](https://aclanthology.org/W15-1509/), which have been included in our repository in the data folder ' ./data '.

## Requirements
* python==3.8
* pytorch==1.12.0
* transformers==4.26.1
* scipy==1.10.1
* numpy==1.23.5
* scikit-learn==1.2.1

## Running
Pre-training, training and testing our model through the bash scripts:
```
sh run.sh
```

## Thanks
Some code references the following repositories:
* [DeepAligned](https://github.com/thuiar/DeepAligned-Clustering)

## Citation
