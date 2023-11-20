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

## Process and dimension
<p align="center">
  <img src="./image/weidu.png" style="transform: scale(0.5);" alt="猫图片">
</p>

## 3. Method
### Overall Structure
We provide a detailed architecture of the  Attention Representation Alignment Fusion network framework we propose for cross-modal information, which comprises three main modules: ARAF encoder module, Representation Alignment module, and Representation Enhancement module. Note that the model uses three different types of cross-modal information including image, text, and time series information in diverse downstream tasks.



<div style="text-align: center;">
<img src="./image/image1.jpg" width="730px">
</div>


###  Compatible Attention Encoder
The ARAF Encoder is a unified component for cross-modal feature extraction. This encoder leverages attention mechanisms and the Modality Fusion Enhancement (MFE) module to better focus on relevant portions of cross-modal representation information. This structure effectively transforms information from different modalities into a common sequence data format, unifying the representation of diverse modal data, and thereby reducing heterogeneity and representation differences among different modalities.



<div style="text-align: center;">
<img src="./image/encoder.png" width="650px">
</div>


## 4.  Experiment
In this section, we evaluate and visualize classification, regression, and generation tasks separately.
### Classification

<p align="center">
  <img src="./image/vqa.png" width="900" height="420", scale= 0.5, alt="猫图片">
</p>
This experiment focuses on five categories of prediction tasks related to Visual Question Answering (VQA). It is a comparative study aimed at illustrating the differences in prediction results when using the ARAF framework compared to not using it. The experimental results are presented through ROC (Receiver Operating Characteristic) curve graphs. From the experimental results, we observed that networks utilizing the ARAF framework exhibit superior performance as indicated by their larger ROC curve areas. This suggests that the ARAF framework has a positive impact on enhancing the performance of VQA tasks.

### Regression

<div style="text-align: center;">
<img src="./image/huigui-1.png" width="730px">
</div>
To visualize the effectiveness of the Electricity Transformer Dataset (ETD) predictive regression task, especially the efficacy of the Adaptive Fusion Framework (ARAF) fusion modality, an image can be created containing two scatterplots. The first scatterplot illustrates the relationship between predicted results and actual labels when the ARAF framework is used, with points distributed along the diagonal to demonstrate the proximity of predictions to actual values. The second scatterplot is a residual scatterplot when employing the ARAF framework, which displays the differences between predicted values and true values. Such a visual presentation can assist in comprehending and evaluating the performance of the ARAF framework in handling the ETD prediction task.

### Generation
<div style="text-align: center;">
<img src="./image/bb-1.png" width="730px">
</div>

<div style="text-align: center;">
<img src="./image/xiaoguo-1.png" width="730px">
</div>

The CUB dataset can be employed to generate an image illustrating the experimental outcomes, showcasing the effectiveness of the ARAF framework in image generation tasks. This image showcases numerous bird images generated using the ARAF framework, which combines visual attributes and semantic information as inputs. These images serve as evidence that the ARAF framework proficiently produces high-quality, vivid, and semantically coherent bird images.

## Alignment and Fusion Mechanism Visualization
<p align="center">
  <img src="./image/test.png" width="900" height="420", scale= 0.5, alt="猫图片">
</p>

<p align="center">
  <img src="./image/test1.png" width="900" height="420", scale= 0.5, alt="猫图片">
</p>

To visualize the distribution of data in 3D space and compare the changes before and after using the ARAF framework, we can design an image. This image include four 3D scatterplots, each showing the data distribution at different stages:

1. The top-left plot displays the data distribution without using the ARAF framework.
2. The top-right plot shows the data space distribution after both modalities have been processed through our unified attention encoder.
3. The bottom-left plot illustrates the data distribution after using the feature alignment fusion module.
4. The bottom-right plot showcases the data space distribution after the data has gone through the representation enhancement module.

The 3D scatterplots clearly demonstrated the positions of data points in space and emphasized how the ARAF framework effectively reduced the common semantic space of the data after its use. This visualization helped illustrate how the ARAF framework, by optimizing the spatial distribution of data, improved the network's predictive performance in downstream tasks. Through a direct comparison of these four plots, the optimization effect of the ARAF framework could be observed clearly.
 

##  Comparison of ARAF with other Fusion Methods

<p align="center">
  <img src="./image/com.png" style="transform: scale(0.5);" alt="猫图片">
</p>

## 4. Training and evaluation
Train &  Test: run Train_Test.py












