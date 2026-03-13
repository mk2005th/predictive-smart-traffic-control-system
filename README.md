🚦 Predictive Smart Traffic Control System

Overview

The Predictive Smart Traffic Control System is an intelligent traffic management solution that dynamically adjusts traffic signal timings based on real-time vehicle detection and predicted traffic density. The system uses computer vision and machine learning techniques to analyze traffic videos, detect vehicles, estimate traffic density, and predict upcoming traffic flow. Based on these predictions, the system automatically allocates optimal signal timings to reduce congestion and improve traffic efficiency.
This project demonstrates the integration of YOLOv8 object detection with Echo State Networks (ESN) for predictive traffic management.

Problem Statement
Traditional traffic signals operate using fixed timers, which do not adapt to real-time traffic conditions. This often results in:
Long waiting times
Traffic congestion
Inefficient road usage
Increased fuel consumption
The goal of this project is to build a smart adaptive system that dynamically adjusts signal timings according to traffic density.

Objectives
Detect vehicles from traffic video using computer vision
Count vehicles to estimate traffic density
Predict traffic flow using machine learning
Dynamically allocate green signal timing
Reduce congestion and waiting time

Technologies Used
Python -> Core programming
YOLOv8	-> Vehicle detection
OpenCV	-> Video processing
NumPy	-> Data processing
Echo State Network	-> Traffic prediction
CSV Dataset	-> Training traffic data

System Architecture
Traffic Video Input
        │
        ▼
Vehicle Detection (YOLOv8)
        │
        ▼
Vehicle Counting
        │
        ▼
Traffic Density Calculation
        │
        ▼
Echo State Network Model
(Traffic Prediction)
        │
        ▼
Dynamic Signal Timing Allocation
        │
        ▼
Optimized Traffic Flow

Installation
1. Clone Repository
git clone https://github.com/yourusername/predictive-smart-traffic-control-system.git
cd predictive-smart-traffic-control-system
2. Install Dependencies
pip install -r requirements.txt

Running the Project
Step 1: Collect Traffic Data
python collect_data.py
Step 2: Train Prediction Model
python train_esn.py
Step 3: Run Smart Traffic System
python run_system.py

Output
The system will:
Detect vehicles from traffic video
Count vehicles per frame
Predict traffic density
Dynamically allocate signal time

Results
Reduced waiting time at intersections
Adaptive traffic signal timing
Improved traffic flow efficiency

<img width="763" height="799" alt="Screenshot 2026-03-13 091505" src="https://github.com/user-attachments/assets/c771b441-629c-400a-8ff4-e2c4a8259f2d" />
<img width="1425" height="839" alt="Screenshot 2026-03-13 091444" src="https://github.com/user-attachments/assets/48edd831-b9b8-4045-9a9b-e7e8edcbe08a" />
