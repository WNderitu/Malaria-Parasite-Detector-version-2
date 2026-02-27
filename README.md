# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8n Model version 2

PROJECT
---
![Version](https://img.shields.io/badge/version-2-blue)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%203.0-lightgrey?style=flat)

TECH STACK & FRAMEWORK
---
![Google_Colab](https://img.shields.io/badge/Platform-Google_Colab-ea580c)
![Google_Drive](https://img.shields.io/badge/Storage-Google_Drive-ea580c)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Model](https://img.shields.io/badge/YOLOv8-Nano-00BFFF?style=flat)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-purple?style=flat)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![YOLOv8n](https://img.shields.io/badge/Model-YOLOv8n-4338ca)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-4338ca)
![ONNX](https://img.shields.io/badge/Export-ONNX-6b7280)
![ONNX_Runtime](https://img.shields.io/badge/Runtime-ONNX_Runtime_CPU-6b7280)
![PyTorch](https://img.shields.io/badge/Deep_Learning-PyTorch-ee4c2c)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-16a34a)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-0d9488)
![Altair](https://img.shields.io/badge/Altair-Interactive_Charts-7c3aed)
![Pillow](https://img.shields.io/badge/Pillow-Image_Processing-db2777)
![PyYAML](https://img.shields.io/badge/PyYAML-Config-b45309)
![imagehash](https://img.shields.io/badge/imagehash-Deduplication-6b7280)
![ReportLab](https://img.shields.io/badge/ReportLab-PDF_Generation-dc2626)
![BBBC041](https://img.shields.io/badge/Dataset-BBBC041_Broad_Institute-7c3aed)

This repository is an update on a previous repository https://github.com/WNderitu/Malaria-parasite-P.Vivax-detector-and-counter.git. In this repository, 1031 images are utilised in model training. 

## Introduction

### Malaria

A life-threatening disease, caused by Plasmodium parasites transmitted by female Anopheles mosquitoes. Five types of plasmodium parasites cause malaria; _Plasmodium Vivax_ (focus of this project), _Plasmodium Falciparum_, _Plasmodium Ovale_, _Plasmodium Malariae_ and _Plasmodium Knowelsi_. Malaria parasites undergo different lifecycle stages namely: hypnozoite, merozoite, ring, trophozoite, schizont and gametocyte while in human blood and tissues. Four lifecycle stages may be detected in infected human red blood cells during a parasites life cycle in man.  Globally, there were 282 million new cases(1 out of 16 people developed malaria) and 610,000 deaths (13.8 deaths per 100,000 people at risk).  In africa, There were 265 million new cases (94% of cases globally) which translates to 1 out of 4 people developed malaria (237.6 cases per 1,000 people at risk). In terms of deaths, Africa had 579,000 deaths (95% of deaths globally) which was also quantified as 51.9 deaths per 100,000 people at risk. 75% of all deaths in Africa occured in children less than 5 years of age. This statistics show that Africa had a heavy burden of malaria especially in Sub Saharan Africa.(Source: WHO Global Malaria Report, 2025)

### Malaria Diagnosis and Need for Advanced Detection Systems

The gold standard test for the diagnosis of malaria is microscopy. The test involves examining and detecting malaria parasite in a patient's blood sample. The microscopist identifies the type of parasite, the different lifecycle stages & the parasite quantity/load. In the laboratory, a Giemsa-stained thick blood smear is prepared from the patient's blood sample by a microscopist. This process is normally manual, time consuming and requires a skilled and expert microscopist. In some settings there is a shortage of laboratory personell with the required skills. In addititon, ensuring expert slide preparation and reading can be difficult and in some rural settings, microscopy is often unavailable. 

It is important for there to be accurate and quick detection of parasites in patient's blood smears so as to have prompt treatment and prevention of disease relapse especially in the case of malaria caused by Plasmodium Vivax parasite. Therefore there is a need for advanced detection systems such as artificial intelligent aided systems. This systems may improve improve reliability and efficiency in identifying malaria parasites causing malaria infections.

## Problem Statement

The detection of Plasmodium vivax malaria remains challenging, primarily due to the morphological similarities between parasite lifecycle stages in infected red blood cells and other blood components observed in microscopic images. Conventional microscopy is a labor-intensive process that is susceptible to human error and demands specialized expertise, which may be limited in under-resourced settings. Consequently, there is an urgent need for automated, efficient, and dependable solutions capable of detecting and quantifying Plasmodium vivax parasites and their developmental stages within blood smears. Such advancements would enhance diagnostic accuracy, promote effective treatment, and strengthen malaria control initiatives.

## Proposed Solution

A You Only Look Once (YOLO) v8n model for detecting infected red blood cells and the parasite stage in the red blood cells and counting the number of parasite stages per image. This model leverages its advanced capabilities to accurately identify and localize multiple object types within images or video frames. A diagram illustrating the solution is shown. The model will be trained on images. 

<img width="1381" height="677" alt="image" src="https://github.com/user-attachments/assets/3ff66565-0bcd-46ae-b2c5-a8db50e0a5c7" />

## Project Objectives

1.	To develop a computer vision model for object detection and counting
2.	To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
3.	To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear using the developed YOLOv8 model
4.	To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
5.	 To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image using the developed YOLOv8 model

## Image Dataset
![Dataset](https://img.shields.io/badge/Dataset-BBBC041-orange?style=flat)
![Images](https://img.shields.io/badge/Total_Images-1%2C328-7c3aed)
![Resolution](https://img.shields.io/badge/Resolution-1600_×_1200-7c3aed)
![Format](https://img.shields.io/badge/Format-JPG_/_PNG-6b7280)

Microscopic images obtained from Broad Bioimage Benchmark Collection website <https://bbbc.broadinstitute.org/BBBC041/>. The dataset consists of an image folder, training json file & test json file. Sample images from the dataset are shown. 

<img width="491" height="325" alt="image" src="https://github.com/user-attachments/assets/24c25e2a-f8b3-47c6-837b-bb9d800e3430" />

<img width="447" height="319" alt="image" src="https://github.com/user-attachments/assets/dfdab58d-827d-496b-8da6-bcee41ffbd42" />

## Deep learning Model Architecture
![Model](https://img.shields.io/badge/YOLOv8-Nano-00BFFF?style=flat)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-purple?style=flat)

The YOLO model is a single stage detector that predicts bounding boxes and class probabilities directly from the entire input image in a single forward pass, which makes the model faster than other object detection models. The model treats object detection as a single regression problem. YOLOv8 will be used for object detection and counting. The model size to be used is yolov8n (Nano) which has about 3 million parameters, is the fastest, suitable for small datasets and computers with limited GPU. However, it's accuracy is lower than other bigger sizes of YOLOv8 models. 

The model is dividied into three main components:
- **Backbone (feature extractor)** - this consists of the CNN that is responsible for extracting hierarchical features from the input image.
- **Neck** - this merges/fuses feature maps from the different stages of the backbone to capture information at various scales.
- **Head** - this is responsible for making predictions. It takes the merged features from the neck and outputs bounding box coordinates, class probabilities, and confidence scores for detected objects. The Head typically consists of multiple detection heads, each connected to a different output scale from the Neck, enabling the prediction of objects at various sizes. Post-processing techniques like non-maximum suppression (NMS) are applied to filter out redundant or overlapping bounding box predictions, resulting in the final set of detected objects.

<img width="805" height="749" alt="image" src="https://github.com/user-attachments/assets/285f3df0-be30-4a7f-8071-302d6f66cd20" />

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

### 1.0 Data Preparation
![Notebook](https://img.shields.io/badge/Notebook-1_·_Data_Preparation-6b7280)
![Libraries](https://img.shields.io/badge/Libraries-Pillow_·_NumPy_·_Pandas_·_Matplotlib_·_PyYAML_·_imagehash-0d9488)
![Classes](https://img.shields.io/badge/Classes-7-db2777)

#### 1.1 Checking for Data Imbalance
The blood smears contain red blood cells infected with _Plasmodium vivax_ parasite. For each image, a class label and set of bounding box coordinates are given. There are 7 Class labels: red blood cell (uninfected), trophozoite, gametocyte, schizont, difficult, ring & leukocyte (uninfected). The Red Blood Cell and Leukocyte classes are blood cells that are not infected with the malaria parasite. The Trophozoite, Schizont, Ring and Gametocyte are different growth stages of the malaria parasite. The difficult class label is for observed parasite growth stages that couldn't be grouped into either of the 4 growth classes. 
The number of images in training subset was 1208 & 120 images for the test subset. Class Imbalance noted in the train, val and test image subsets as shown in the charts below. The imbalance is severe with 96% of the objects being from the red blood cell class. This imbalance is inherent to human blood smears as they have more red blood cells than other cells found in blood.

<img width="692" height="460" alt="image" src="https://github.com/user-attachments/assets/871445a8-f0d1-4e60-888f-abece37b9220" />

#### 1.2 Handling Data Imbalance
![Balancing](https://img.shields.io/badge/Class_Balancing-RBC--only_removed-16a34a)

This involved selecting from the 1208 training images, images with only 'Red Blood Cell' annotations, removing them, inorder to reduce class imbalance. 177 images were removed and moved to a separate folder. 1031 images remained for use in model training. The corresponsing training JSON file was updated. 

<img width="700" height="490" alt="image" src="https://github.com/user-attachments/assets/406cee7b-f6f9-4d87-a92e-1634c82ba685" />

#### 1.3 Dataset preparation for YOLOv8n model
ANNOTATIONS
![Conversion](https://img.shields.io/badge/JSON_→_YOLO-Annotation_Conversion-16a34a)
![Labels](https://img.shields.io/badge/Label_Format-YOLO_TXT-4338ca)
![Config](https://img.shields.io/badge/Config-data.yaml-0d9488)

DATASPLIT
![Train](https://img.shields.io/badge/Train-825_images-16a34a)
![Validation](https://img.shields.io/badge/Validation-206_images-b45309)
![Test](https://img.shields.io/badge/Test-120_images-2563eb)
![Split](https://img.shields.io/badge/Split-Train_/_Val_/_Test-16a34a)

Involved converting training and test JSON files to YOLOv8 txt format, creation of class weights, yolov8_malaria dataset folder with image & labels subfolders, test, training & val folders in images directory and val folder in labels directory. Some of the training images were moved to the val image and label subfolders from the training folders to have a val folder with 206 images. The resulting training set folders (image & labels folder) were updated to result in 825 images. Lastly a data configuration file appropriate for YOLOv8n model training was created.
### Class weights

<img width="496" height="465" alt="image" src="https://github.com/user-attachments/assets/6288a34d-7372-4c68-83e9-5bf8dfeb31c6" />

### Data configuration 

<img width="658" height="226" alt="image" src="https://github.com/user-attachments/assets/a72cdf43-12c5-4891-8235-8d71b41cb6aa" />

### 2.0 Model Training & Evaluation - Train 1
![Notebook](https://img.shields.io/badge/Notebook-2_·_Model_Training-6b7280)
![Device](https://img.shields.io/badge/Device-GPU_CUDA-76b900)
![Optimizer](https://img.shields.io/badge/Optimizer-SGD_with_Cosine_LR-ea580c)
![LR](https://img.shields.io/badge/Learning_Rate-0.001_→_0.0001-b45309)
![Epochs](https://img.shields.io/badge/Epochs-500-orange?style=flat-square)
![Patience](https://img.shields.io/badge/Early_Stopping-Patience_50-6b7280)
![Class_Weights](https://img.shields.io/badge/Class_Weights-Custom_7_classes-7c3aed)

![mAP50](https://img.shields.io/badge/mAP50-0.743-16a34a)
![mAP50_95](https://img.shields.io/badge/mAP50--95-0.581-16a34a)
![Precision](https://img.shields.io/badge/Precision-0.665-0d9488)
![Recall](https://img.shields.io/badge/Recall-0.740-0d9488)

The prepared dataset was used for training the YOLOv8n model. Number of epochs was 500, with image size of 1280 and a batch size of 8. Augmentations for microscopy were implemented such as small object augmentations, colour augmentations and geometric augmentations. Class loss and box loss was also implemented. This first training was interrupted 3 times due to GPU timeout, adding of aggressive learning rate decay and no improvement to finally end training at epoch 354/500. 

<img width="643" height="373" alt="image" src="https://github.com/user-attachments/assets/cdc872a5-9df9-4f4a-837b-ce79b986c570" />

<img width="633" height="355" alt="image" src="https://github.com/user-attachments/assets/0e114ffa-e686-4836-8cce-820f1b3119a0" />

<img width="624" height="362" alt="image" src="https://github.com/user-attachments/assets/a48b99db-fe11-4c93-b369-4d869745870c" />

<img width="616" height="350" alt="image" src="https://github.com/user-attachments/assets/db917443-ac03-4dc6-9b14-dd6d3addec7f" />

<img width="623" height="354" alt="image" src="https://github.com/user-attachments/assets/2272bd84-3cf6-42eb-bac6-4e2a534e4758" />

Performance metrics

<img width="739" height="177" alt="image" src="https://github.com/user-attachments/assets/2a84c075-c63d-4799-8b56-e9bd0021ce6b" />

### 3.0 Data Preparation for fine tuning 
![Notebook](https://img.shields.io/badge/Notebook-3_·_Data_Prep_for_Finetune-6b7280)
![Action](https://img.shields.io/badge/Action-Difficult_→_Trophozoite_Merge-dc2626)
![Analysis](https://img.shields.io/badge/Analysis-Co--occurrence_%26_BBox_Dimensions-7c3aed)
![Difficult_Train](https://img.shields.io/badge/Difficult_Class_(Train)-2.45%25_of_annotations-b45309)
![Difficult_Val](https://img.shields.io/badge/Difficult_Class_(Val)-0.68%25_of_annotations-b45309)
![Overlap](https://img.shields.io/badge/Trophozoite_Overlap-133_of_270_images-ea580c)
![Classes](https://img.shields.io/badge/Classes_After_Merge-6_(from_7)-16a34a)
![Output](https://img.shields.io/badge/Output-yolov8_malaria_finetune_dataset-2563eb)

Based on the resulting metrics for precision, recall, mAP50, mAP50-95 and per class metrics, fine tuning of the model was planned. Before fine tuning could be done, the dataset was further processed to try and improve the metrics. This involved removing the difficult class entirely as it was confusing the model and merging with the trophozoite class. 

### 4.0 Hyperparameter tuning (notebook: 4_hyperparametertune.ipynb)
![Notebook](https://img.shields.io/badge/Notebook-4_·_Hyperparameter_Tuning-6b7280)
![Strategy](https://img.shields.io/badge/Strategy-Fine--tune_from_best.pt-4338ca)
![Augmentation](https://img.shields.io/badge/Augmentation-Extended_Mosaic-0d9488)
![Scale](https://img.shields.io/badge/Scale_Augmentation-0.7_(from_0.5)-0d9488)
![Epochs](https://img.shields.io/badge/Epochs-200_max-6b7280)
![LR](https://img.shields.io/badge/LR-Aggressive_Decay_Cosine-b45309)

![mAP50_v1](https://img.shields.io/badge/Finetune_v1_mAP50-0.798-16a34a)
![mAP50_95_v1](https://img.shields.io/badge/Finetune_v1_mAP50--95-0.624-16a34a)
![Precision_v1](https://img.shields.io/badge/Avg_Precision-0.756-0d9488)
![Recall_v1](https://img.shields.io/badge/Avg_Recall-0.759-0d9488)
![Best_Epoch](https://img.shields.io/badge/Best_Epoch-131_(mAP50_0.804)-db2777)

Hyperparameter tuning was done twice (Fine Tune 1 & Fine Tune 2). Based on the resulting metrics, the best.pt was chosen from fine tune 1 model for evaluation on 120 test images and deployment. The model had significantly better Recall (fewer missed detections) for common (Trophozoite) and rare (Gametocyte) classes, the accuracy was marginally higher and performed best on detecting the different parasite classes. 

<img width="2202" height="470" alt="image" src="https://github.com/user-attachments/assets/936e259c-85c3-46a3-be3f-0329fca0f150" />

### Curves
<img width="575" height="327" alt="image" src="https://github.com/user-attachments/assets/b4bb803b-b800-4fd4-9993-112b4a6211db" />

<img width="525" height="299" alt="image" src="https://github.com/user-attachments/assets/5f100a5b-38bc-47f8-9628-f5ab9901dd99" />

<img width="606" height="345" alt="image" src="https://github.com/user-attachments/assets/7a02ddc1-98bc-4e34-959a-a71554b5226d" />

<img width="527" height="297" alt="image" src="https://github.com/user-attachments/assets/74a42669-8a28-4994-89b7-b307cd19fa2f" />

<img width="520" height="336" alt="image" src="https://github.com/user-attachments/assets/902916b6-20de-4200-a796-4656b898a987" />

<img width="577" height="411" alt="image" src="https://github.com/user-attachments/assets/61b2e700-27cf-4859-b9e3-655eb378fc89" />

### 4.0 Model Evaluation on Test Images
![Notebook](https://img.shields.io/badge/Notebook-5_·_Model_Evaluation-6b7280)
![Weights](https://img.shields.io/badge/Weights-best.pt_(Finetune_v1)-4338ca)
![Test_Images](https://img.shields.io/badge/Test_Images-120-7c3aed)
![Conf](https://img.shields.io/badge/Confidence_Threshold-0.05-ea580c)
![IoU](https://img.shields.io/badge/IoU_NMS-0.45-ea580c)
![Classes](https://img.shields.io/badge/Classes-6_(RBC·Leukocyte·Schizont·Ring·Trophozoite·Gametocyte)-db2777)
![Export](https://img.shields.io/badge/Export-ONNX-2563eb)
![mAP50](https://img.shields.io/badge/mAP50-0.400-b45309)
![mAP50_95](https://img.shields.io/badge/mAP50--95-0.301-b45309)
![F1](https://img.shields.io/badge/F1--Score-0.391-b45309)
![RBC_Precision](https://img.shields.io/badge/RBC_Precision-90.0%25-16a34a)
![RBC_Recall](https://img.shields.io/badge/RBC_Recall-95.2%25-16a34a)
![Trophozoite_Precision](https://img.shields.io/badge/Trophozoite_Precision-57.0%25-0d9488)
![Ring_Precision](https://img.shields.io/badge/Ring_Precision-46.0%25-ea580c)
![Schizont_Precision](https://img.shields.io/badge/Schizont_Precision-18.1%25-dc2626)

Model did not perform as expected due to the inherently imbalanced microscopic image dataset. A microscopic image of a blood smear will always have red blood cells in abundance which makes the model have few examples of parasite classes to learn effectively from.  

<img width="1660" height="451" alt="image" src="https://github.com/user-attachments/assets/cb4a8535-1591-4825-8e27-d6487e1d690f" />

Overall, the model had poor performance in detecting all classes. In regard to per class performance, the model had good performance in detecting red blood cell class and was heavily biased on detecting classes as being red blood cells. Poor performance was noted in detecting the 3 rare classes; schizont, ring and gametocyte classes. Moderate performance was noted in detecting the trophozoite class. 

### 5.0 Deployment
![App](https://img.shields.io/badge/App-Malaria_Parasite_Detector_v2-dc2626)
![Framework](https://img.shields.io/badge/Framework-Streamlit-ff4b4b)
![Model](https://img.shields.io/badge/Model-YOLOv8n_ONNX-4338ca)
![Runtime](https://img.shields.io/badge/Runtime-ONNX_Runtime_CPU-6b7280)
![Version](https://img.shields.io/badge/Version-2.0-16a34a)
![Input](https://img.shields.io/badge/Input_Resolution-1280_×_1280-7c3aed)
![Parasitemia](https://img.shields.io/badge/Parasitemia-WHO_Classification-2563eb)
![Export](https://img.shields.io/badge/Export-CSV_·_PDF_·_ZIP_Images-0d9488)
![Use_Case](https://img.shields.io/badge/Use_Case-Research_Only_(Not_Clinical)-6b7280)

The selected model was deployed on streamlit app. The app can be accessed at https://malaria-parasite-detector-version-2-kuhwhgejcrbgyxsma4pzrp.streamlit.app/

## Recommendations/Future work

1. Use larger YOLOv8 model e.g. medium variant
2. Develop 2 stages for detecting & classifying malaria parasites
  - 1st stage: detect uninfected red blood cells vs. infected red blood cells with YOLOv8 model
  - 2nd stage: classify infected red blood cells into various parasite stages with an image classifier model e.g. AlexNet, EfficientNet, GoogLeNet, ResNet , MobileNet, Vision Transformers (ViT) etc..
3. Expand model to detect other types of malaria – multi type malaria parasite detector

## Acknowledgements & Attributions
1. We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
2. Image of YOLOv8 model architecture from: https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5

## References
1. https://www.who.int/health-topics/malaria#tab=tab_1
2. https://www.who.int/teams/global-malaria-programme/reports/world-malaria-report-2025
3. https://docs.ultralytics.com/datasets/
4. https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5]
5. Link to data: Broad Bioimage Benchmark Collection website https://bbbc.broadinstitute.org/BBBC041/
