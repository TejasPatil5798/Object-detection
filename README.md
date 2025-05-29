# 🎯 Real-Time Object Detection and Tracking with YOLOv8

This project showcases real-time object detection and tracking using the YOLOv8 model (Ultralytics). It supports detecting objects from webcam streams, static images, and recorded video files using OpenCV.

## 📁 Project Structure


├── object_tracking.py        # Real-time detection using webcam

├── OD image.py               # One-time detection on a captured webcam image

├── OD recorded vid.py        # Object detection on a video file

├── yolov8n.pt                # YOLOv8n model weights (not included, add manually)

├── requirements.txt          # Python dependencies


## 🚀 Features
Live object detection and tracking via webcam

Image-based single-shot object detection

Object detection on recorded video files with output video saving

Bounding boxes with class ID and confidence overlay

Lightweight YOLOv8n model for real-time performance


## 🧰 Requirements
Python 3.7+

opencv-python-headless

ultralytics

Install all dependencies with:


pip install -r requirements.txt


## 📦 YOLOv8 Model
Download the YOLOv8n weights and place the yolov8n.pt file in the project directory.

## ▶️ Usage
1. Real-Time Object Tracking (Webcam)

python object_tracking.py


2. Image Capture and Detection (Webcam snapshot)


python "OD image.py"


3. Object Detection on a Video File
Edit the video path in OD recorded vid.py and run:

python "OD recorded vid.py"
Press q to stop any running OpenCV window.


## 📸 Output
Bounding boxes with class ID and confidence scores

Output video file generated as output_video.mp4 (for recorded video script)

Output demo video is attached.

## Made with 💡 and OpenCV + YOLOv8
