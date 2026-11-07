# Soccer Ball Tracking – Computer Vision Miniproject

This repository contains the code and trained model developed for my Computer Vision miniproject. The objective was to detect and track a soccer ball using two different approaches:

1. Traditional computer vision.
2. A neural network-based approach using YOLOv8.

The two methods were evaluated on a separate one-minute soccer video provided through Moodle.

## Repository Structure

```text
Miniprojekt_Submission/
│
├── Code/
│   ├── traditional_detector_codex.py
│   ├── evaluate_traditional.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── args.yaml
│   └── README.txt
│
├── Model/
│   └── best.pt
│
├── Annotations/
│   └── evaluation_annotations.zip
│
└── Report/
    └── report.pdf
```

## 1. Traditional Computer Vision

The traditional method uses image processing techniques to identify potential soccer ball candidates.

The implementation consists of:

- HSV colour thresholding to identify bright regions.
- Region of interest (ROI) selection.
- Contour detection and geometric filtering.
- Candidate scoring based on visual properties.
- A simple constant-velocity tracker to follow the ball between frames.

**Main file:** `traditional_detector_codex.py`

The evaluation script compares the detected ball positions with the manually annotated ground truth.

**Evaluation file:** `evaluate_traditional.py`

## 2. Neural Network – YOLOv8

The neural network approach uses a pretrained YOLOv8n model, which was fine-tuned to detect a single class: the soccer ball.

The model was trained using the SoccerNet dataset:

- Training data: Games 4 and 6.
- Validation data: Game 9.
- Training images: 4,976.
- Validation images: 2,316.

Every fifth frame was sampled from the original videos.

**Training file:** `train_yolo.py`

The trained model weights are included in:

`Model/best.pt`

The training configuration is also included in `Code/args.yaml`.

The evaluation script loads the trained weights and evaluates the model on the independent Moodle video.

**Evaluation file:** `evaluate_yolo.py`

## 3. Test Data and Annotations

The final evaluation was performed on a one-minute soccer video provided through Moodle.

The video was manually annotated using CVAT (Computer Vision Annotation Tool). Every fifth frame was annotated, resulting in 300 evaluation frames.

The exported annotations are included in:

`Annotations/evaluation_annotations.zip`

The same annotations were used to evaluate both methods.

The original test video is not included because it was provided through Moodle.

## 4. Requirements

The project was developed using:

- Python 3.11.9
- OpenCV
- NumPy
- PyYAML
- Ultralytics YOLOv8
- PyTorch with CUDA support

The main Python packages can be installed using:

```bash
pip install opencv-python numpy pyyaml ultralytics==8.3.67
```

A compatible NVIDIA GPU and CUDA-enabled PyTorch installation are recommended for training YOLO.

## 5. Running the Code

Run the scripts from the project directory after ensuring that the required datasets, annotations and model weights are available at the paths expected by the scripts.

To train the YOLO model:

```bash
python Code/train_yolo.py
```

To evaluate YOLO:

```bash
python Code/evaluate_yolo.py
```

To evaluate the traditional computer vision method:

```bash
python Code/evaluate_traditional.py
```

Note that the original SoccerNet training data and Moodle evaluation video are not included in this repository. They must be obtained separately if the scripts are to be run again from the beginning.

## 6. Evaluation Results

The final evaluation produced the following results:

| Metric | YOLOv8n | Traditional CV |
|:---|---:|---:|
| Precision | 32.2% | 2.7% |
| Recall | 50.5% | 3.8% |
| F1-score | 39.3% | 3.1% |

YOLOv8n performed considerably better than the traditional computer vision implementation on the independent evaluation video.

For further details about the methods, implementation and results, please refer to the project report.
