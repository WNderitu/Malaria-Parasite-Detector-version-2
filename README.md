# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8n Model version 2

This repository is an update on a previous repository https://github.com/WNderitu/Malaria-parasite-P.Vivax-detector-and-counter.git. In this repository, 1031 images utilised in model training. 

## Introduction

### Malaria
- A life-threatening disease, caused by Plasmodium parasites transmitted by female Anopheles mosquitoes and is most commonly found in tropical and subtropical regions.
- Five types of plasmodium parasites cause malaria;
  -   _Plasmodium Vivax_ (focus of this project)
  -   _Plasmodium Falciparum_
  -   _Plasmodium Ovale_
  -   _Plasmodium Malariae_
  -   _Plasmodium Knowelsi_

Malaria parasites undergo different lifecycle stages namely: hypnozoite, merozoite, ring, trophozoite, schizont and gametocyte while in human blood and tissues. Four lifecycle stages may be detected in infected human red blood cells during a parasites life cycle in man. 

### Malaria Disease Burden in 2024

Globally, there were 282 million new cases(1 out of 16 people developed malaria) and 610,000 deaths (13.8 deaths per 100,000 people at risk).  In africa, There were 265 million new cases (94% of cases globally) which translates to 1 out of 4 people developed malaria (237.6 cases per 1,000 people at risk). In terms of deaths, Africa had 579,000 deaths (95% of deaths globally) which was also quantified as 51.9 deaths per 100,000 people at risk. 75% of all deaths in Africa occured in children less than 5 years of age. This statistics show that
Africa had a heavy burden of malaria especially in Sub Saharan Africa.(Source: WHO Global Malaria Report, 2025)

### Malaria Diagnosis and Need for Advanced Detection Systems

The gold standard test for the diagnosis of malaria is microscopy. The test involves examining and detecting malaria parasite in a patient’s blood sample. The microscopist identifies the type of parasite, the different lifecycle stages & the parasite quantity/load. In the laboratory, a Giemsa-stained thick blood smear is prepared from the patient's blood sample by a microscopist. This process is normally manual, time consuming and requires a skilled and expert microscopist. In some settings there is a shortage of laboratory personell with the required skills. In addititon, ensuring expert slide preparation and reading can be difficult and in some rural settings, microscopy is often unavailable. 

It is important for there to be accurate and quick detection of parasites in patient's blood smears so as to have prompt treatment and prevention of disease relapse especially in the case of malaria caused by Plasmodium Vivax parasite. Therefore there is a need for advanced detection systems such as artificial intelligent aided systems. This systems may improve improve reliability and efficiency in identifying malaria parasites causing malaria infections.

## Problem Statement

The detection of Plasmodium vivax malaria remains challenging, primarily due to the morphological similarities between parasite lifecycle stages in infected red blood cells and other blood components observed in microscopic images. Conventional microscopy is a labor-intensive process that is susceptible to human error and demands specialized expertise, which may be limited in under-resourced settings. Consequently, there is an urgent need for automated, efficient, and dependable solutions capable of detecting and quantifying Plasmodium vivax parasites and their developmental stages within blood smears. Such advancements would enhance diagnostic accuracy, promote effective treatment, and strengthen malaria control initiatives.

## Project Objectives

1.	To develop a computer vision model for object detection and counting
2.	To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
3.	To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear using the developed YOLOv8 model
4.	To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
5.	 To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image using the developed YOLOv8 model

## Image Dataset

Images obtained from Broad Bioimage Benchmark Collection website <https://bbbc.broadinstitute.org/BBBC041/>. The dataset consists of an image folder, training json file & test json file. There are 1,328 microscopic images of blood smears with a resolution of 600x1200. The blood smears contain red blood cells infected with _Plasmodium vivax_ parasite. For each image, a class label and set of bounding box coordinates are given. There are 7 Class labels: red blood cell (uninfected), trophozoite, gametocyte, schizont, difficult, ring & leukocyte (uninfected). The Red Blood Cell and Leukocyte classes are blood cells that are not infected with the malaria parasite. The Trophozoite, Schizont, Ring and Gametocyte are different growth stages of the malaria parasite. The difficult class label is for observed parasite growth stages that couldn't be grouped into either of the 4 growth classes. Sample images from the dataset are shown. 

<img width="491" height="325" alt="image" src="https://github.com/user-attachments/assets/24c25e2a-f8b3-47c6-837b-bb9d800e3430" />

<img width="447" height="319" alt="image" src="https://github.com/user-attachments/assets/dfdab58d-827d-496b-8da6-bcee41ffbd42" />

## Project Description
This project employs YOLOv8n variant for multiclass object detection, leveraging its advanced capabilities to accurately identify and localize multiple object types within images or video frames.

## Deep learning Model Architecture

The You Only Look Once (YOLO) model is a single stage detector that predicts bounding boxes and class probabilities directly from the entire input image in a single forward pass, which makes the model faster than other object detection models. The model treats object detection as a single regression problem. 

The YOLO version 8 model (YOLOv8) will be used for object detection and counting. The model size to be used is yolov8n (Nano) which has about 3 million parameters, is the fastest, suitable for small datasets and computers with limited GPU. However, it's accuracy is lower than other bigger sizes of YOLOv8 models. 

The model is dividied into three main components:
- **Backbone (feature extractor)** - this consists of the CNN that is responsible for extracting hierarchical features from the input image.
- **Neck** - this merges/fuses feature maps from the different stages of the backbone to capture information at various scales.
- **Head** - this is responsible for making predictions. It takes the merged features from the neck and outputs bounding box coordinates, class probabilities, and confidence scores for detected objects. The Head typically consists of multiple detection heads, each connected to a different output scale from the Neck, enabling the prediction of objects at various sizes. Post-processing techniques like non-maximum suppression (NMS) are applied to filter out redundant or overlapping bounding box predictions, resulting in the final set of detected objects.

<img width="1207" height="1122" alt="image" src="https://github.com/user-attachments/assets/4665efe1-8dd4-4cbc-b2f1-d57c7475b34c" />

**Object detection evaluation metrics** used will be precision, recall, F1 Score and mean average precision (mAP).
  
- **Precision**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of positive detections (True Positives + False Positives).It tells you how accurate the model is when it predicts an object is present. High precision = fewer false detections.
  
Precision = TP / (TP + FP)

- **Recall**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of actual positive objects in the image (True Positives + False Negatives). It tells you how many of the actual objects the model was able to find. High recall = fewer missed detections.

Recall = TP / (TP + FN)

- **F1 Score**: Harmonic mean of precision and recall.YOLOv8 often reports best F1 (at optimal confidence threshold).

<img width="214" height="35" alt="Screenshot 2025-11-02 at 17 57 44" src="https://github.com/user-attachments/assets/293697af-cd78-4848-9c66-97ac24540aca" />

- **mAP@0.5** — IoU threshold = 0.5 (i.e., boxes overlap ≥ 50% to count as correct) - mean average precision calculated at a fixed IOU threshold of 0.50. This generally assesses whether the model can generaly detect the presence and approximate location of an object, and is a less less stric metric. 
 
- **mAP@0.5:0.95** — Mean mAP across IoU thresholds 0.5 to 0.95 (step 0.05) - average of the mean average precision calculated across multiple IoU thresholds, ranging from 0.50 to 0.95 in steps of 0.05 (i.e 0.50, 0.55, 0.60,...,0.95). 

Other metrics to help understand mAP@0.5 & mAP@0.5-0.95 performance metrics in YOLOv8:
- **Intersection over Union (IoU)**: This measures the overlap between the model's predicted bounding box and the actual ground truth bounding box. An IoU of 1 means perfect overlap, while 0 means no overlap. A common threshold (e.g., 0.5) is set to consider a detection as a True Positive. Higher IoU = better localization accuracy. 

<img width="172" height="38" alt="Screenshot 2025-11-02 at 17 58 54" src="https://github.com/user-attachments/assets/7a788e3c-8038-46ee-90b7-c98468378cb3" />

## Methodology

### 1.0 Data Preparation (notebook:1_data_preparation.ipynb)

#### 1.1 Checking for Data Imbalance

The number of images in training subset was 1208 & 120 images for the test subset. Class Imbalance noted in the train, val and test image subsets as shown in the charts below. The imbalance is severe with 96% of the objects being from the red blood cell class. This imbalance is inherent to human blood smears as they have more red blood cells than other cells found in blood.

TRAINING set:
difficult: 441 (0.55%)
gametocyte: 144 (0.18%)
leukocyte: 103 (0.13%)
red blood cell: 77420 (96.64%)
ring: 353 (0.44%)
schizont: 179 (0.22%)
trophozoite: 1473 (1.84%)

 TEST set:
difficult: 5 (0.08%)
gametocyte: 12 (0.20%)
leukocyte: 0 (0.00%)
red blood cell: 5614 (94.80%)
ring: 169 (2.85%)
schizont: 11 (0.19%)
trophozoite: 111 (1.87%)

<img width="887" height="590" alt="image" src="https://github.com/user-attachments/assets/404e5401-ec87-44b4-b29a-b254e9dcb110" /> 

#### 1.2 Handling Data Imbalance

This involved selecting from the 1208 training images, images with only 'Red Blood Cell' annotations, removing them, inorder to reduce class imbalance. 177 images were removed and moved to a separate folder. 1031 images remained for use in model training. The corresponsing training JSON file was updated. 

Resulting distribution was:
TRAINING set (excluding red blood cell only images):
difficult: 441 (0.64%)
gametocyte: 144 (0.21%)
leukocyte: 103 (0.15%)
red blood cell: 65721 (96.06%)
ring: 353 (0.52%)
schizont: 179 (0.26%)
trophozoite: 1473 (2.15%)

TEST set:
difficult: 5 (0.08%)
gametocyte: 12 (0.20%)
leukocyte: 0 (0.00%)
red blood cell: 5614 (94.80%)
ring: 169 (2.85%)
schizont: 11 (0.19%)
trophozoite: 111 (1.87%)

<img width="986" height="690" alt="image" src="https://github.com/user-attachments/assets/9e291480-af8c-4e92-b0cd-528911fd6229" />

#### 1.3 Dataset preparation for YOLOv8n model
The following steps were done:
i. Training and test JSON files were converted to YOLOv8 txt format.
ii. Creation of class weights
iii. Creation of yolov8_malaria dataset folder with image & labels subfolders
iv. Creation of test, training & val folders in images directory
v. Creation of val folder in labels directory
vi. Moving of some training images to the val image and label subfolders from the training folders to have a val folder with 206 images. 
vii. Updating of the resulting training set folders  (image & labels folder) to result in 825 images. 
viii. Creation of a data configuration file appropriate for YOLOv8n model training.

<img width="842" height="491" alt="Screenshot 2026-02-23 192115" src="https://github.com/user-attachments/assets/ae11f4c9-d8de-4e89-ae5c-21d7424b75a9" />

### 2.0 Model Training & Evaluation - Train 1

The prepared dataset was used for training the YOLOv8n model (see notebook:2_model_train.ipynb). Image size used for training was 1280 with a batch size of 8, 500 epochs with early stop at 300 and a learning rate of 0.001. Augmentations for microscopy were implemented such as small object augmentations, colour augmentations and geometric augmentations. Class weights were also applied. Class loss and box loss was also implemented. This first training was interrupted 3 times due to GPU timeout, adding of aggressive learning rate decay andmno improvement to finally end training at epoch 354/500. 

<img width="678" height="393" alt="image" src="https://github.com/user-attachments/assets/fac14782-cd50-4ac0-aa5b-21dfff5299aa" />

<img width="700" height="393" alt="image" src="https://github.com/user-attachments/assets/8e5e7789-6e87-4aa3-b351-a6ad6e2ec26b" />

<img width="678" height="393" alt="image" src="https://github.com/user-attachments/assets/4302d664-ec98-48ca-b32f-58d5b2977079" />

<img width="678" height="393" alt="image" src="https://github.com/user-attachments/assets/0a8d364d-e153-435c-90e8-4ad8eb4a9a57" />

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/04dcd04e-9949-456c-b5d6-e4a3aa076ae3" />

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/76eef77c-ae7d-43de-9700-80d5ba048da0" />


<img width="1254" height="301" alt="image" src="https://github.com/user-attachments/assets/2a316fb6-92d9-41bd-8bfc-446c8ef9fffa" />

Based on the resulting metrics for precision, recall, mAP50, mAP50-95 and per class metrics, fine tuning of the model was planned. Before fine tuning could be done, the dataset was further processed to try and improve the metrics (see notebook: 3_data_preparation_for_finetune.ipynb). This involved removing the difficult class entirely as it was confusing the model and merging with the trophozoite class. 

### 3.0 Hyperparameter tuning (notebook: 4_hyperparametertune.ipynb)
Hyperparameter tuning was done twice (Fine Tune 1 & Fine Tune 2). Based on the resulting metrics, fine tune 1 model was chosen for evaluation on 120 test images and deployment. The model had significantly better Recall (fewer missed detections) for common (Trophozoite) and rare (Gametocyte) classes, the accuracy was marginally higher and performed best on detecting the different parasite classes. The metrics are shown. 

<img width="1632" height="403" alt="image" src="https://github.com/user-attachments/assets/6add65e4-9c36-42b8-9361-470ce457bc1b" />

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/2df855fe-2a3f-43d6-be97-ed593c3e7306" />

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/ae5b6239-3a3b-40e2-ae6b-7494ef9adb9a" />

<img width="691" height="393" alt="image" src="https://github.com/user-attachments/assets/998861f9-e4d1-44ff-910e-2fd7766dcf5d" />

<img width="699" height="393" alt="image" src="https://github.com/user-attachments/assets/b643cc9a-098f-477e-accf-cd0c867b3485" />

<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/27de94b9-6f1b-4080-a26b-423f81a33625" />

### Model Evaluation
### 5.0 Model Evaluation on Test Images
### Predicting on test images
### Model Selection
## Results
## Deployment
https://malaria-parasite-detector-version-2-kuhwhgejcrbgyxsma4pzrp.streamlit.app/

## Recommendations/Future work
1. Use larger YOLOv8 model e.g. medium variant
2. Use 2 stages for detecting & classifying malaria parasites
  - 1st stage: detect uninfected red blood cells vs. infected red blood cells with YOLOv8 model
  - 2nd stage: classify infected red blood cells into various parasite stages with an image classifier model e.g. AlexNet, EfficientNet, GoogLeNet, ResNet , MobileNet, Vision Transformers (ViT) etc..

## Acknowledgements & Attributions
1. We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
2. Image of YOLOv8 model architecture from: https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5
3. Image of P.Vivax lifecylce. Quique Bassat, CC BY 4.0 <https://creativecommons.org/licenses/by/4.0>, via Wikimedia Commons

## References
1. https://www.who.int/health-topics/malaria#tab=tab_1
2. https://docs.ultralytics.com/datasets/
3. https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5]
4. Link to data: Broad Bioimage Benchmark Collection website https://bbbc.broadinstitute.org/BBBC041/


