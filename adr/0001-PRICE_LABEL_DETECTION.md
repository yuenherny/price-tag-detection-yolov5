# PROPOSAL: Price Label Detection

## Status
Proposed by: Yu Yuen Hern (1 Nov 2022)

Updated by: Yu Yuen Hern (8 Nov 2022)

## Context
Given an image of a supermarket shelf, the we need to:
1. Develop a model that knows where the position of price label(s) are in the image
2. Save them as JPG for further processing (OCR).

Note: There are no training datasets given. 

## Proposed Approach
Based on this [paper](https://www.researchgate.net/publication/359213858_Neural_Network-Based_Price_Tag_Data_Analysis), the most performing model for price tag detection is Yolov4-Tiny. However we will be using YOLOv5  in this project since there were no official PyTorch implementation for YOLOv4.

The YOLOv5 that we use is pretrained on the COCO dataset.

### Approach 1: Domain-adaptation using given images
1. Label the price tags in given image.
2. Modify the output layer to detect 1 class only (price tag) and its coordinates.
3. Retrain the model using labelled data.
4. Evaluate the model using given data.

### Approach 2: Domain-adapt using external images
1. Check if the labels given in [this](https://retailvisionworkshop.github.io/pricing_challenge_2021/) and [this](https://www.kaggle.com/datasets/manikchitralwar/webmarket-dataset) dataset aligns with our objective.
2. If not, label 20 images with price tags.
3. Modify the output layer to detect 1 class only (price tag) and its coordinates.
4. Retrain the model using labelled data.
5. Evaluate the model using given data.

## Consequences

### Using Approach 1
Advantages: Minimal data annotation, not time consuming

Disadvantages: Model is not robust and might perform poorly

### Using Approach 2
Advantages: Model is robust and performs better

Disadvantages: Time consuming, extensive data annotation, does not adhere to Agile approach

## Discussion

### A. Checking if labels given in [Retail Vision dataset](https://retailvisionworkshop.github.io/pricing_challenge_2021/) aligns with our objective
The given labels are product annotations, not price tags annotations.

### B. Checking if labels given in [Webmarket dataset](https://www.kaggle.com/datasets/manikchitralwar/webmarket-dataset) aligns with our objective
The given labels are product annotations, not price tags annotations.