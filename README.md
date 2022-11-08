# Price Tag Detection using YOLOv5

This repo is forked and modified from [Ultralytics's YOLOv5 Pytorch implementation](https://github.com/ultralytics/yolov5). Visit here for the original [README](https://github.com/ultralytics/yolov5#readme) and [Training Custom Data tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

## Table of Content
1. [Project structure](#project-structure)
2. [How to use](#how-to-use)
3. [Results](#results)
4. [Reflection and Future Work](#reflection-and-future-work)

## Project structure
The project is structured as follows:
```
<root>
|-- adr (architectural decision records)
|-- archive
|-- classify (classification)
|-- data
|   |-- hyps (hyperparam YAMLs)
|   |   |-- hyp.*.yaml
|   |   |-- ...
|   |
|   |-- images (images folder)
|   |-- scripts (bash scripts folder)
|   |-- pricetag.yaml (dataset config for this project)
|   |-- *.yaml (configs for other datasets)
|
|-- datasets
|   |-- pricetag
|       |-- images (training images)
|       |   |-- *.jpg
|       |   |-- ...
|       |   
|       |-- labels (labels in YOLO format)
|           |-- *.txt
|           |-- ...
|
|-- models (model configs)
|   |-- hub
|   |-- segment
|   |-- *.py
|   |-- yolov5*.yaml (YOLOv5 configs)
|
|-- runs
|   |-- detect
|   |   |-- exp* (folder containing detection results)
|   |
|   |-- train
|       |-- exp* (folder containing training results)
|
|-- segment (Python files for segmentation)
|-- utils (utility functions)
|-- detect.py (run this to perform inference)
|-- train.py (run this to train your model)
|-- *.py (other Python files)
|-- requirements.txt
```
Note: Some files are omitted in the project tree above for the sake of brevity.

## How to use
You can take the following steps to replicate the project:
1. Clone this repo using Git Bash.
    ```
    $ git clone https://github.com/yuenherny/price-tag-detection-yolov5.git
    ```
2. Create a virtual environment.
    ```
    $ cd price-tag-detection-yolov5
    $ python -m venv venv
    $ source venv/Scripts/activate
    $ pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
    $ python install -r requirements.txt
    ```
3. If you are using any existing datasets like COCO or Pascal VOC, please follow this [tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#3-train). I used [Label Studio](https://labelstud.io/) to label my own data. Put your custom data anywhere at the root as you need to specify the path to the data later.
4. Start the training.
    ```
    $ python train.py --data pricetag.yaml --weights yolov5s.pt --img 640 --batch-size 1
    ```
    **Note that batch size is 16 by default. You may change it to speed up the training process or to avoid memory issues.**
5. You will get some output in the console.
    ```
    ...
        Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        98/99      3.02G    0.09491     0.1833          0         41        640: 100%|██████████| 6/6 [00:01<00:00,  3.25it/s]
                    Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 3/3 [00:00<00:00,  7.60it/s]
                    all          6        189     0.0144      0.138     0.0118    0.00303
        Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
        99/99      3.02G    0.09637     0.1838          0         44        640: 100%|██████████| 6/6 [00:01<00:00,  3.25it/s]
                    Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 3/3 [00:00<00:00,  7.61it/s]
                    all          6        189     0.0144      0.138     0.0118    0.00303

    100 epochs completed in 0.158 hours.
    Optimizer stripped from runs\train\exp11\weights\last.pt, 173.0MB
    Optimizer stripped from runs\train\exp11\weights\best.pt, 173.0MB

    Validating runs\train\exp11\weights\best.pt...
    Fusing layers...
    Model summary: 322 layers, 86173414 parameters, 0 gradients, 203.8 GFLOPs
                    Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 3/3 [00:00<00:00,  5.43it/s]
                    all          6        189     0.0172      0.164     0.0141    0.00279
    Results saved to runs\train\exp11
    ```
6. Perform inference using the same data.
    ```
    $ python detect.py --weights path/to/best_model.pt --source path/to/images
    ```
7. You will get some output in the console.
    ```
    detect: weights=['runs/train/exp22/weights/best.pt'], source=datasets/pricetag/images, data=data\coco128.yaml, imgsz=[640, 640], conf_thres=0.9, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs\detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1      
    YOLOv5  v6.2-234-g1b7ffe7 Python-3.10.5 torch-1.13.0+cu116 CUDA:0 (NVIDIA GeForce RTX 2060, 6144MiB)

    Fusing layers...
    Model summary: 322 layers, 86173414 parameters, 0 gradients, 203.8 GFLOPs
    image 1/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\40e49c68-Task_1-1.jpg: 640x480 27 pricetags, 51.0ms
    image 2/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\42f0f230-Task_1-2.jpg: 640x480 21 pricetags, 45.0ms
    image 3/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\66ab95ea-Task_2-2.jpg: 640x480 40 pricetags, 45.0ms
    image 4/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\a0a3cd09-Task_1-3.jpg: 640x480 21 pricetags, 46.0ms
    image 5/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\b4df6ee6-Task_2-3.jpg: 640x480 73 pricetags, 46.0ms
    image 6/6 D:\Repos\GitHub\price-tag-detection-yolov5\datasets\pricetag\images\b6c4a491-Task_2-1.jpg: 640x480 21 pricetags, 50.0ms
    Speed: 1.5ms pre-process, 47.2ms inference, 10.2ms NMS per image at shape (1, 3, 640, 640)
    Results saved to runs\detect\exp12
    ```

## Results
Note: Results here are as of 8th Nov 2022. Results may be updated from time to time.

### Best model training config:
- Model: YOLOv5x
- Freeze layers: 0 to 9 (YOLO backbone=layer 10)
- Optimizer: Adam
- Image size: 640px
- Batch size: 1
- Epochs: 100
- Earlystopping: 10 epochs without improvement
- Other hyperparam config: See `./data/hyps/hyp.scratch.yaml`

### Best model inference config:
- Confidence threshold: 0.90
- NMS threshold: 0.45

### Performance Metric
- Precision: 0.053096
- Recall:  0.29101
- MAP50: 0.038881
- MAP50-95: 0.0075325

### Inferencing

<img src="/results/exp12/40e49c68-Task_1-1.jpg" alt="Image 1" width="640">

For more results, see `./results/exp12` folder.

## Reflection and Future Work
The current project status did not achieve the following intended objectives:
1. Find all the price tag in the image given and sort the text data on each price tag in a table format.
2. Find and group all the products in the image by brand, then output the data in table format.

It only partially achieves the first part of Objective 1: Detecting price tags in images before cropping them out for OCR processing.

However, the inference results were not satisfactory as expected:
1. The performance metric was exceptionally bad at MAP50 0.039.
2. In the inference output, the model appears to be able to detect a number of price tags in the images, but there were also false positives especially at the lower area of each rack level.
3. The model was also equally confident on true and false positives, indicating that it is still unable to differentiate true positives from false positives.
4. Due to time limitation, only 6 images was labelled. Hence, the approach was to try out hyperparam tuning. Unfortunately, the results were not satisfactory despite using the **Extra Large variant of YOLOv5 with high data augmentation and only freezing backbone YOLO layers**.
5. Hence it was suspected that lack of labelled data could be the main issue. Future work could focus on annotating more data.