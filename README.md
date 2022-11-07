# Price Tag Detection using YOLOv5

This repo is forked and modified from [Ultralytics's YOLOv5 Pytorch implementation](https://github.com/ultralytics/yolov5). Visit here for the original [README](https://github.com/ultralytics/yolov5#readme) and [Training Custom Data tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).

## How to use
You can take the following steps to replicate the project:
1. Clone this repo using Git Bash.

    `$ git clone https://github.com/yuenherny/price-tag-detection-yolov5.git`

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

    `$ python train_pricetag.py --data pricetag.yaml --weights yolov5s.pt --img 640 --batch-size 1`

    **Note that batch size is 16 by default. You may change it to speed up the training process or to avoid memory issues.**
5. Perform inference using the same data.

    `$ python detect_pricetag.py --weights path/to/best_model.pt --source path/to/images`

6. You will get some output in the console.
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