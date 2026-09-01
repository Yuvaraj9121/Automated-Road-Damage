# 🛣️ Automated Road Damage Detection using UAV Images

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.0-red?logo=keras)](https://keras.io/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)](https://ultralytics.com/)

## 📝 Overview
This research-oriented project implements an **Automated Road Damage Detection System** utilizing **UAV (Unmanned Aerial Vehicle)** imagery and state-of-the-art **Deep Learning** techniques. The system provides a graphical interface to compare performance across multiple YOLO architectures and perform real-time damage detection.

The project has been fully modernized and upgraded to **Python 3.12.0**:
- **Core Engine**: Migrated to **TensorFlow 2.16.1** and **Keras 3.0**.
- **YOLO Integration**: Full support for `ultralytics` YOLOv8.
- **Performance**: Optimized for Python 3.12 with enhanced memory management.
- **UI Modernization**: Updated Tkinter interface logic for better stability.

## 🛠️ Technology Stack
- **Languages**: Python 3.12.0
- **Deep Learning**: TensorFlow, Keras, Ultralytics (YOLOv8)
- **Computer Vision**: OpenCV (OpenCV-Python, OpenCV-Contrib)
- **Data Analysis**: NumPy, Pandas, Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **GUI**: Tkinter

## 📋 Prerequisites
Ensure you have the following installed:
- [Python 3.12.0+](https://www.python.org/downloads/)
- [Git LFS](https://git-lfs.github.com/) (required for downloading large model files)

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/srohithadithya/Automated_road-damage.git
   cd Automated_road-damage
   ```

2. **Setup Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r RoadDamage/requirements.txt
   ```

## 🖥️ Usage

1. **Navigate to the Source Directory**:
   ```bash
   cd RoadDamage
   ```

2. **Launch the Application**:
   - Double-click `run.bat`
   - **OR** run via terminal:
     ```bash
     python Main.py
     ```

## 🔍 Features & Workflow

### 1. Data Management
- **Upload Dataset**: Load the RDD2022 (China Drone) dataset annotations and images.
- **Preprocess Dataset**: Normalize coordinates and preview processed samples.
- **Split Dataset**: Automated 80/20 train-test split for model evaluation.

### 2. Model Evaluation
Compare performance metrics (Accuracy, Precision, Recall, F1-Score) across:
- **YOLOv5** (Legacy Baseline)
- **YOLOv7** (Proposed Enhancement)
- **YOLOv8** (State-of-the-Art Extension)

### 3. Real-Time Detection
- **Upload Test Images**: Run inference on new UAV images to detect road damage (e.g., D00, D10, D20, D40) or repairs in real-time.

## 📊 Comparison Results
The system generates a dynamic **Confusion Matrix** and a **Comparison Graph** to visualize the superiority of the YOLOv8 extension over previous versions.

## 📁 Project Structure
```text
Automated_road-damage/
├── RoadDamage/
│   ├── Main.py              # Main Application Entry Point
│   ├── requirements.txt     # Python Dependencies
│   ├── run.bat              # Windows Launch Script
│   ├── model/               # Models & Weights (.hdf5, .pt, .npy)
│   ├── testImages/          # Sample Images for Inference
│   └── RDD2022_China_Drone/ # Dataset Structure
├── .gitignore               # Project Exclusions
└── README.md                # Project Documentation
```

## 🧪 Research Context
This project was developed to evaluate the efficiency of UAVs in automated infrastructure monitoring. By comparing **YOLOv5**, **YOLOv7**, and **YOLOv8**, the research demonstrates how evolutionary improvements in deep learning architectures contribute to higher mAP (mean Average Precision) in detecting complex road distress patterns under varying lighting and altitude conditions.

---
**Created for research and educational purposes in Automated Infrastructure Monitoring.**
