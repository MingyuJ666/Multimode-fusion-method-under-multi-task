# Attention-based cross-modal representation alignment and fusion mechanism for multitasks

Cross-modal representation alignment is a critical mechanism for integrating multi-modal data. This technique involves the iterative transmission and enhancement of feature representations between different modalities to effectively integrate multi-modal information and generate a unified representation in the feature space. However, existing methods often struggle with feature dimension mismatches, representation space inconsistencies, and lack of adaptability to diverse downstream tasks. To address these challenges, we propose Attention-based Representation Alignment Fusion (ARAF), a promising approach that is highly compatible and robust in capturing cross-modal representation information. It enhances model performance by implementing a consistent alignment fusion mechanism. The proposed method outperforms other popular multi-modal fusion approaches in various tasks, including regression, classification, and image generation. This superiority is consistently demonstrated through extensive empirical evaluations on multiple datasets. The results consistently indicate that the Attention Representation Alignment Fusion method achieves state-of-the-art performance in these tasks. 




## 1. Requirements
- PyTorch >= 1.10.1
- python >= 3.7
- Einops = 0.6.1
- numpy = 1.24.3 
- torchvision = 0.9.1+cu111 
- scikit-learn = 1.2.2  
- CUDA >= 11.3

## 2. Data preparation
We use the [VQA](https://visualqa.org/vqa_v1_download.html) as classification dataset.

We use the [ETDdataset](https://github.com/zhouhaoyi/ETDataset) as regression dataset.

We use the [CUB](https://paperswithcode.com/dataset/cub-200-2011) as image generation dataset.

We use these three datasets to test the multimodal fusion performance.

## 3. Method
### Overall Structure
We will provide a detailed architecture of the  Attention Representation Alignment Fusion network framework we propose for cross-modal information, which comprises three main modules: ARAF encoder module, Representation Alignment module, and Representation Enhancement module. Note that the model uses three different types of cross-modal information including image, text, and time series information in diverse downstream tasks.



<div style="text-align: center;">
<img src="./image/image1.jpg" width="730px">
</div>


###  Compatible Attention Encoder
The ARAF Encoder is a unified component for cross-modal feature extraction. This encoder leverages attention mechanisms and the Modality Fusion Enhancement (MFE) module to better focus on relevant portions of cross-modal representation information. This structure effectively transforms information from different modalities into a common sequence data format, unifying the representation of diverse modal data, and thereby reducing heterogeneity and representation differences among different modalities.



<div style="text-align: center;">
<img src="./image/image2.jpg" width="450px">
</div>


## 4.  Expeiment
In this section, we evaluate and visualize classification, regression, and generation tasks separately.
### Classification

<p align="center">
  <img src="./image/final2-1.png" width="700" height="480", scale= 0.5, alt="猫图片">
  <img src="./image/cao-1.png" width="600" height="480", scale= 0.5 alt="狗图片">
</p>


### Regression
### Generation



## 4. Training and evaluation
**Train &  Test: run Train_Test.py**









