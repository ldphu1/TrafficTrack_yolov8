# TrafficTrack_yolov8
Here is my python source code for TrafficTrack - an automated vehicle detection, tracking, and counting system. With my code, you could:
* Run an app which you could track and count vehicles from a video file (`.mp4`)
* Run an app which you could track and count vehicles from an image sequence (extracted frames)
  <p align="center">
  <img src="https://github.com/user-attachments/assets/106c1786-6790-4431-bddd-2ffeb851efe1" width="30%"/>
  <img src="https://github.com/user-attachments/assets/15ad69f3-6383-4f3b-8f97-4e9dfa9ad971" width="30%"/>
  <img src="https://github.com/user-attachments/assets/109b8130-ed7f-42a2-bb0b-cab35f3b750f" width="30%"/></p>

# Tracking Apps

In order to use this app, you need a traffic video or a sequence of images. When a vehicle appears in the frame, it will be detected, tracked with a colored bounding box, and its trajectory (tail) will be drawn. A virtual line (Line Crossing) is set up at the bottom of the frame; whenever a vehicle crosses this line, the counter will increase. 
Below are the scripts to run the demo:
* **For video:** simply run `python3 test_vd.py`
* **For image sequence:** simply run `python3 detrac_sequence_test.py`

# Dataset

The dataset used for training my model is the **UA-DETRAC** dataset, which is a challenging real-world traffic dataset. 
Due to its large size, I only uploaded a small sample of image sequences (`sample_sequence/`) and a demo video in the `data/` folder for testing purposes. You can download the full original dataset from their official website.

# Categories

The table below shows 4 vehicle categories my model used for classification:

| car | bus | van | others |
|:---:|:---:|:---:|:---:|

# Trained models

Due to GitHub's file size limits, the trained weights are hosted externally. 
 **[Download best.pt here](https://drive.google.com/file/d/1wFvfnLZDDCHhaULvct_feOG76z7I7t39/view?usp=drive_link)**

*Note: After downloading, please place the `best.pt` file inside the `weights/` folder before running any scripts.*

# Training

You need to download the UA-DETRAC Images and XML annotation files and store them in your local folder. 
1. First, you need to convert the XML annotations to YOLO format by configuring the input paths and running the script:
   `python3 format_data.py`
   This script will automatically parse XMLs, convert bounding boxes, split train/val sets, and generate the `data.yaml` file.
2. If you want to train your model with a different list of categories, you only need to change the constant `CLASS` dictionary at `format_data.py`.
3. Then you could simply run YOLOv8 training using the generated `data.yaml`.

# Experiments
<p align = "center">
<img width="700" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/f03d9256-8019-4816-8c0b-06fd02398892" />
</p>

The model was trained using the `yolov8s.pt` pre-trained weights. I trained the model for 50 epochs. The model reached its convergence point and achieved impressive tracking performance on highway scenarios. The key metrics for the best epoch are shown below:
* **mAP@50:** ~80.0%
* **mAP@50-95:** 65.9%

# Requirements

* python 3.8+
* ultralytics (YOLOv8)
* opencv-python (cv2)
* numpy
