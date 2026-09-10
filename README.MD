# 🚧 Automated Road Damage Detection Using UAV Images

A computer-vision and deep-learning project for detecting road damage from **UAV (Unmanned Aerial Vehicle) images**. The project provides a desktop interface for dataset preprocessing, model training/evaluation, performance comparison, and image-based road-damage detection.

## 📌 Overview

Road surface damage such as cracks and other forms of deterioration can affect transportation safety and maintenance planning. Manual inspection of large road networks is time-consuming and difficult to scale.

This project explores an automated approach using UAV imagery and deep-learning-based image analysis. The application processes annotated road images, prepares the data for model training, evaluates multiple model implementations, and provides an image inference workflow for detecting road damage.

The project includes implementations/workflows referred to in the application as:

* YOLOv5
* YOLOv7
* YOLOv8 extension/inference workflow

> **Note:** The YOLOv5 and YOLOv7 components in this repository are custom neural-network implementations using Keras/TensorFlow rather than the official Ultralytics YOLOv5/YOLOv7 implementations.

---

## 🎯 Objectives

* Automate road-damage analysis from UAV images.
* Process XML-based road-damage annotations.
* Normalize bounding-box coordinates.
* Split the dataset into training and testing sets.
* Train/evaluate multiple deep-learning approaches.
* Generate classification metrics and confusion matrices.
* Perform road-damage detection on new images.
* Provide a simple desktop GUI for the complete workflow.

---

## ✨ Features

### Dataset Processing

* Load annotated road images.
* Read XML annotations.
* Extract road-damage class labels.
* Extract and normalize bounding boxes.
* Prepare NumPy arrays for model processing.
* Create an 80/20 train-test split.

### Model Evaluation

The application contains workflows for:

* Custom YOLOv5-style model
* Custom YOLOv7-style model using MobileNetV2 as a backbone
* YOLOv8-based detection/inference workflow

The evaluation code calculates:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Image Detection

The application can load a test image and use the available YOLOv8 model weights to identify road-damage regions.

### Desktop Interface

The project uses **Tkinter** to provide buttons for:

* Upload Dataset
* Preprocess Dataset
* Split Dataset
* Run YOLOv5
* Run YOLOv7
* Run YOLOv8
* Generate comparison graph
* Perform damage detection

---

## 🗂️ Dataset

The repository contains the annotation portion of the **RDD2022 China Drone** dataset structure.

The project expects the dataset to be organized approximately as:

```text
RDD2022_China_Drone/
├── annotations/
│   └── xmls/
│       ├── China_Drone_000000.xml
│       ├── China_Drone_000001.xml
│       └── ...
└── images/
    ├── China_Drone_000000.jpg
    ├── China_Drone_000001.jpg
    └── ...
```

The supplied project contains **2,401 XML annotation files**.

The annotations contain the following road-related labels:

| Class         | Description                                               |
| ------------- | --------------------------------------------------------- |
| `D00`         | Longitudinal crack                                        |
| `D10`         | Transverse crack                                          |
| `D20`         | Alligator/other crack category represented by the dataset |
| `D40`         | Pothole                                                   |
| `Repair`      | Repaired road surface                                     |
| `Block crack` | Present in a small number of annotations                  |

The current application primarily defines the five classes:

```text
D00
D10
D20
D40
Repair
```

> **Dataset note:** The repository archive does not include the corresponding image files, so the complete image dataset needs to be supplied separately according to the expected directory structure.

---

## 🔄 Project Workflow

```text
UAV Road Images
       ↓
XML Annotations
       ↓
Extract Classes & Bounding Boxes
       ↓
Normalize Bounding Boxes
       ↓
Dataset Preparation
       ↓
80/20 Train-Test Split
       ↓
┌──────────────┬──────────────┬──────────────┐
│ Custom       │ Custom       │ YOLOv8       │
│ YOLOv5       │ YOLOv7       │ Workflow     │
└──────────────┴──────────────┴──────────────┘
       ↓
Model Prediction
       ↓
Accuracy / Precision / Recall / F1
       ↓
Confusion Matrix & Comparison
       ↓
Road-Damage Detection on New Images
```

---

## 🧠 Model Approaches

### YOLOv5 Workflow

The project implements a Keras/TensorFlow convolutional neural network with:

* Convolution layers
* Max-pooling
* Fully connected layers
* Bounding-box regression output
* Classification output

The model is trained for up to **20 epochs** when the corresponding saved weights are not available.

### YOLOv7 Workflow

The project uses **MobileNetV2** as a feature-extraction backbone followed by separate heads for:

* Bounding-box prediction
* Class prediction

Dropout and dense layers are used in the classification branch.

The model is trained for up to **20 epochs** when weights are not already available.

### YOLOv8 Workflow

The project supports loading a YOLOv8-compatible model through the **Ultralytics** package.

For image detection, the application expects:

```text
RoadDamage/model/best.pt
```

The detection workflow uses a confidence threshold of approximately **0.30** when deciding whether a detected region should be highlighted.

---

## Evaluation Results

The detector was evaluated on a separate test split that was not used for model selection or training.

### Test Set

| Metric | Result |
|---|---:|
| Test images | 288 |
| Ground-truth boxes | 466 |
| Precision | 34.24% |
| Recall | 30.85% |
| mAP@50 | 25.12% |
| mAP@50–95 | 11.90% |

These results were generated by running the project's `evaluate_detector.py` evaluation script against the test dataset.

The reported metrics are object-detection metrics. They should not be interpreted as classification accuracy.

### Important Limitation

The current model is a research/demo detector and its measured test performance is limited. Predictions can be incorrect, particularly for difficult road conditions, image quality variations, or damage types that are underrepresented in the training data.

The detector should therefore be treated as an assistive machine-learning system rather than a definitive road-condition assessment tool.
---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **Ultralytics**
* **OpenCV**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Tkinter**

---

## 💻 Requirements

Recommended environment:

* Python 3.12+
* Windows/Linux/macOS
* Sufficient RAM and storage for the dataset and model files

Main Python dependencies are listed in:

```text
RoadDamage/requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Yuvaraj9121/Automated-Road-Damage.git
```

### 2. Enter the project directory

```bash
cd Automated-Road-Damage
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
.\venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r RoadDamage/requirements.txt
```

---

## ▶️ Running the Application

Move into the application directory:

```bash
cd RoadDamage
```

Run:

```bash
python Main.py
```

On Windows, if the repository contains the launch script, you can also use:

```text
run.bat
```

---

## 🖥️ Using the Application

### Step 1 — Upload Dataset

Use:

```text
Upload Dataset
```

Select the directory containing the road-damage dataset.

### Step 2 — Preprocess Dataset

Use:

```text
Preprocess Dataset
```

The application displays an example image with its processed bounding box.

### Step 3 — Split Dataset

Use:

```text
Splitting Dataset
```

The code uses an **80/20 train-test split** with `random_state=42`.

### Step 4 — Run Model

Select one of:

```text
Run Existing YOLOV5 Algorithm
Run Proposed YOLOV7 Algorithm
Run Extension YOLOV8 Algorithm
```

### Step 5 — Compare Results

Use:

```text
Run Comparison Graph
```

to visualize the recorded evaluation metrics.

### Step 6 — Detect Road Damage

Use:

```text
Damage Detection
```

and select a road image.

If `model/best.pt` is available, the YOLOv8 detection workflow processes the image and displays detected regions.

---

## 📁 Project Structure

```text
Automated-Road-Damage/
│
├── RoadDamage/
│   ├── Main.py
│   ├── requirements.txt
│   ├── run.bat
│   │
│   ├── RDD2022_China_Drone/
│   │   └── annotations/
│   │       └── xmls/
│   │
│   └── testImages/
│       ├── 1.jpg
│       ├── 2.jpg
│       ├── 3.jpg
│       └── ...
│
├── .gitignore
└── README.md
```

Model files and the complete image dataset may need to be obtained/generated separately depending on the repository version.

---

## 🖼️ Screenshots / Demo

Add screenshots of the running application here.

Recommended screenshots:

1. Main application window
2. Dataset preprocessing result
3. Road-damage detection result
4. Confusion matrix
5. Model comparison graph

Example:

```markdown
## Screenshots

### Main Application
![Main Application](screenshots/main-window.png)

### Road Damage Detection
![Road Damage Detection](screenshots/damage-detection.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion-matrix.png)

### Model Comparison
![Model Comparison](screenshots/model-comparison.png)
```

---

## 🎥 Demo Video

A short demonstration video can show:

```text
1. Launch application
2. Load dataset
3. Preprocess data
4. Split dataset
5. Run model
6. Display evaluation result
7. Upload test image
8. Display road-damage detection
```

Add the video link here:

```text
Demo: [Add your YouTube/Google Drive demo link]
```

---

## ⚠️ Limitations

* The complete image dataset is not included in the repository archive.
* Model weight files may need to be supplied or generated separately.
* The project is primarily a research/educational prototype.
* The current application does not provide a deployed web inference service.
* The evaluation implementation should be independently verified before using the reported metrics as a formal benchmark.
* The current detection display primarily labels detected regions as road damaged/repaired rather than providing a detailed per-class explanation to the user.

---

## 🚀 Future Improvements

* Deploy the detector as a web application.
* Add a REST API for image inference.
* Use a standardized official YOLO training pipeline.
* Add mAP@0.5 and mAP@0.5:0.95 evaluation.
* Provide per-class precision, recall and F1-score.
* Add a proper validation/test protocol.
* Include quantitative model-comparison tables.
* Add more UAV road images from different environments.
* Improve detection of small and overlapping road defects.
* Add automated report generation for road inspection.
* Provide Docker-based reproducible deployment.

---

## 📚 Research / Educational Context

This project demonstrates how computer vision and deep learning can be applied to automated infrastructure inspection using aerial imagery.

The main focus is the processing of road images, extraction of annotated damage information, model-based prediction, and visualization of detection results.

---

## 👤 Author

**Yuvaraj**

GitHub:
https://github.com/Yuvaraj9121

---

## 📄 License

This project is intended for educational and research purposes. Check the licenses of the underlying datasets, libraries, pretrained models, and third-party resources before redistributing them.
