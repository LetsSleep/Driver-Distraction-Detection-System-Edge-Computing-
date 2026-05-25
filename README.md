git # Driver Distraction Detection System (Edge Computing)

## 📖 Overview
This project implements a real-time **Driver Distraction Detection System** designed to run entirely on edge computing hardware. By leveraging deep learning (Keras/TensorFlow) and computer vision (OpenCV), the system monitors driver behavior through local video feeds to detect unsafe actions.

Because the model processes data directly on the edge, it ensures **ultra-low latency**, eliminates the need for a constant internet connection, and completely preserves driver privacy by never transmitting video feeds to the cloud.

## ✨ Key Features
* **Real-Time Inference:** Processes video streams locally via webcam.
* **Behavior Classification:** Accurately identifies specific distractions using a trained Keras model (`keras_Model.h5`).
* **Privacy-First Architecture:** 100% offline edge computing ensures sensitive video data never leaves the vehicle.

## 🛠️ Prerequisites
Ensure you have Python installed on your system. It is recommended to use a virtual environment (`myenv`).

The following Python packages are required:
* `tensorflow` (Required for Keras to work)
* `opencv-python`
* `numpy`

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/AIHT_Proj.git
   cd AIHT_Proj
   ```

2. **Activate your virtual environment:**
   ```bash
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install tensorflow opencv-python numpy
   ```

## 💻 Usage

1. Ensure that your trained model (`keras_Model.h5`) and your labels file (`labels.txt`) are in the root directory of the project.
2. Run the detection script (assuming your script is named `opencv.py`):
   ```bash
   python opencv.py
   ```
3. A window will open displaying your webcam feed with real-time predictions and confidence scores printed to the console.
4. **To exit the application, press the `ESC` key.**

## 📁 Project Structure
* `keras_Model.h5`: The trained deep learning model for classification (Needs to be added).
* `labels.txt`: The text file containing the class names for the model's predictions (Needs to be added).
* `opencv.py` (or your main script): The main computer vision pipeline capturing webcam frames and making predictions.
