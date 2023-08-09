# Temporal CLOCS: Temporal Camera-LiDAR Object Candidates Fusion for 3D Object Detection
Problem: Improve sensor fusion for processing sequential data and automate ground truth generation process


This is the implementation of Master Thesis using CLOCs (Camera-LiDAR Object Candidates Fusion for 3D Object Detection) as the baseline architecture.
The aim of the modified network (Temporal CLOCS) developed during the Master Thesis, is to improve the existing fusion network (CLOCS) by adding the capability for processing sequential data. This is achieved by adding recurrent layers like Convolutional LSTM (Conv-LSTM). They carry forward features from previous time steps to the next. The network is finally trained to detect more accurately the classes and produce 3D and 2D bounding boxes.

Finally, the Temporal CLOCS network was used to automate the generation of bounding boxes, which were used as ground truths to train other deep neural networks developed by the team. The auto ground truth generation process eliminated the need to manually annotate bounding boxes and saved 25% of team working hours.

The Temporal CLOCS architecture developed during the thesis is mentioned below

![CLOCs_LSTM_Architecture](https://github.com/Anirbanbhk88/CLOCS-MasterThesis/assets/2795092/5ed8f2c1-11a1-43e6-9305-e27a77d71df6)
Temporal CLOCS Architecture


## CLASSES detected

Car, Pedestrian, Cyclist

## Sub Networks used inside the Fusion network
2D detector: [Cascade-RCNN](https://github.com/open-mmlab/mmdetection/blob/588536de9905feb7f37c2c977d146a64c74ef28e/mmdet/models/detectors/cascade_rcnn.py#L6)

3D detector: [SECOND](https://github.com/traveller59/second.pytorch/tree/v1.5)

## Environment
Tested on python3.9, pytorch 1.7.1, Ubuntu 20.4/22.4

## Performance on KITTI Object Tracking dataset having sequential data (6711 training, 1297 validation with 85-15 train-val split ratio)
## Temporal CLOCS vs Baseline CLOCS model for Cyclist Class
Baseline CLOCS
```
Cyclist:       Easy@0.5     Moderate@0.5    Hard@0.5
bev AP:         67.57          32.89         30.69
3d  AP:         67.57          32.86         30.57
```

Temporal CLOCS(Ours)
```
Cyclist:       Easy@0.5     Moderate@0.5    Hard@0.5
bev AP:         72.23         36.14          33.71
3d AP:          72.19       36.09          33.68
```

## Temporal CLOCS vs Baseline CLOCS model for Car & Pedestrian Class
## Car
Baseline CLOCS
```
Car:       Easy@0.7     Moderate@0.7    Hard@0.7
bev AP:     96.63​          81.07         75.93​
3d  AP:     95.88​          77.45         72.20​
```

Temporal CLOCS(Ours)
```
Car:       Easy@0.7     Moderate@0.7    Hard@0.7
bev AP:     96.66​         81.29          76.23​
3d  AP:     95.74         77.66          72.56​
```

## Pedestrian
Baseline CLOCS
```
Pedestrian:       Easy@0.7     Moderate@0.7    Hard@0.7
bev AP:            70.05​          67.78         62.95
3d  AP:            65.34​          61.25         56.53
```

Temporal CLOCS(Ours)
```
Pedestrian:       Easy@0.7     Moderate@0.7    Hard@0.7
bev AP:            70.63​         66.79          62.10​
3d  AP:            63.71         60.15          55.56
```



## Future Work
-   Modify the codebase and try with other combinations of 2D and 3D detectors (YOLO, CT3D etc.)
-   Improve further the Car and Pedestrian class
-   

## Installation guide

The code is developed based on SECOND-1.5 for 3D detector, please follow the [SECOND-1.5](https://github.com/traveller59/second.pytorch/tree/v1.5) to setup the environment, the dependences for SECOND-1.5 are needed.
```bash
pip install shapely fire pybind11 tensorboardX protobuf scikit-image numba pillow
```
Follow the instructions to install `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)). The SECOND codebase expects it to be correctly configured.

Then adding the Temporal CLOCs directory to your PYTHONPATH, you could add the following line (change '/dir/to/your/TemporalCLOCs/' according to your Temporal CLOCs directory) in your .bashrc under home directory.
```bash
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/TemporalCLOCs/'
```

## Dataset preparation (KITTI object detection dataset and object tracking dataset)
Download both the KITTI dataset. The object detection dataset is used to replicate the results and pre-train the models. Please follow the instructions mentioned in the baseline CLOCS GitHub repo to arrange the KITTI object detection dataset. 

In order to organize the files for the KITTI object tracking dataset for training the temporal layer in Temporal CLOCS mention the structure as follows:

```plain
└── KITTI_OBJECT_TRACKING_DATASET_ROOT
       ├── training    <-- 6711 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 1297 test data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── kitti_tracking_dbinfos_train.pkl
       ├── kitti_tracking_infos_train.pkl
       ├── kitti_tracking_infos_test.pkl
       ├── kitti_tracking_infos_val.pkl
       └── kitti_tracking_infos_trainval.pkl
```

Next, you could follow the SECOND-1.5 instructions to create kitti infos, reduced point cloud and groundtruth-database infos for the KITTI Tracking dataset. For the info files for KITTI Object detection dataset you can just download these files from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) and put them in the correct directories.


## Fusion of SECOND and Cascade-RCNN
### Preparation
Temporal CLOCs operates on the combined output of a 3D detector and a 2D detector. It then adds ConvLSTM layers to capture sequential data features. For this thesis, we use SECOND as the 3D detector, Cascade-RCNN as the 2D detector. You can try to train any other 2D or 3D detectors of your choice

1. We use detections with sigmoid scores, you could download the Cascade-RCNN detections for the KITTI object detection dataset train and validations set from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) file name:'cascade_rcnn_sigmoid_data'. For training the model with sequential data (KITTI tracking dataset) you have top train the model with the tracking dataset or finetune it.
P.S: However finetuning may result in overfitting. Because the tracking dataset is prepared from same base dataset as the object detection dataset. So similar samples might me there resulting in the model seeing the same data again.

2. Download the pretrained SECOND models from [here](https://drive.google.com/drive/folders/1ScFUWPwzK5_VXb-LYQZuZVkiBj-dTMJ9?usp=sharing) file name: 'second_model.zip', create an empty directory named ```model_dir``` under your CLOCs root directory and unzip the files to ```model_dir```. Your Temporal-CLOCs directory should look like this:

```plain
└── TemporalCLOCs
       ├── d2_detection_data    <-- 2D detection candidates data
       ├── model_dir       <-- SECOND pretrained weights extracted from 'second_model.zip' 
       ├── second 
       ├── torchplus 
       ├── README.md
```
3. Then modify the config file carefully:
```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/dir/to/your/kitti_tracking_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/dir/to/your/kitti_tracking_infos_train.pkl"
  kitti_root_path: "/dir/to/your/KITTI_OBJECT_TRACKING_DATASET_ROOT"
}
...
train_config: {
  ...
  detection_2d_path: "/dir/to/2d_detection/data"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/dir/to/your/kitti_infos_val.pkl"
  kitti_root_path: "/dir/to/your/KITTI_OBJECT_TRACKING_DATASET_ROOT"
}

```
### Train
```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/dir/to/your_model_dir
```
The trained models and related information will be saved in '/dir/to/your_model_dir'


## Deployment
- The saved model is deployed as a Docker container and trained on the Nvidia RTX 3040/3020 GPU 
- CI/CD pipline used: Gitlab CI
