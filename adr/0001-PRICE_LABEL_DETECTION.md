# PROPOSAL: Price Label Detection

## Status
Proposed by: Yu Yuen Hern (1 Nov 2022)

## Context
Given an image of a supermarket shelf, the we need to:
1. Develop a model that knows where the position of price label(s) are in the image
2. Save them as JPG for further processing (OCR).

Note: There are no training datasets given. 

## Proposed Approach
Based on this [paper](https://www.researchgate.net/publication/359213858_Neural_Network-Based_Price_Tag_Data_Analysis), the most performing model for price tag detection is Yolov4-Tiny. Hence we will be using this model in this project.

We make an assumption that Yolov4-Tiny is pretrained on the ImageNet dataset.

### Approach 1: Domain-adaptation using given images
1. Label the price tags in given image.
2. Modify the output layer to detect 1 class only (price tag) and its coordinates.
3. Retrain the model using labelled data.
4. Evaluate the model using given data.

### Approach 2: Domain-adapt using external images
1. Check if the labels given in this [dataset](https://retailvisionworkshop.github.io/pricing_challenge_2021/) are correct.
2. If not, label 20 images with price tags.
3. Modify the output layer to detect 1 class only (price tag) and its coordinates.
4. Retrain the model using labelled data.
5. Evaluate the model using given data.

## Consequences
TBD

## Discussion

### A. Checking if labels given in [dataset](https://retailvisionworkshop.github.io/pricing_challenge_2021/) are correct
The given labels are product annotations, not price tags.