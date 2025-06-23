# Webcam Object Detection with YOLOv8

## Overview

This Python script demonstrates real-time object detection using a webcam feed with the YOLOv8 (You Only Look Once) model from Ultralytics. The program captures video from your default webcam (device 0) and processes each frame to identify and label objects in the scene.

## Key Features

- **Real-time Processing**: Analyzes live webcam feed frame-by-frame
- **Object Detection**: Identifies and classifies multiple objects simultaneously
- **Visual Feedback**: Displays the processed video with bounding boxes and labels
- **Simple Control Flow**: Includes basic logic to potentially exit under certain conditions

## How It Works

1. **Model Initialization**:
   ```python
   model = YOLO('yolo11n.pt')
   ```
   - Loads the YOLOv8 Nano model ('yolo11n.pt'), which is a lightweight version suitable for real-time processing

2. **Webcam Processing**:
   ```python
   results = model(0, show=True)
   ```
   - Captures video from webcam (device 0)
   - Processes each frame through the YOLO model
   - Displays the annotated video feed in real-time (`show=True`)

3. **Result Analysis**:
   ```python
   for result in results:
       boxes = result.boxes
       classes = result.names
   ```
   - Extracts detection boxes (bounding rectangles around detected objects)
   - Gets the class names corresponding to detected objects

4. **Basic Control Logic**:
   ```python
   if boxes > 6:
       break
   ```
   - Includes a simple condition that could terminate processing if many objects are detected
   - (Note: This condition may need adjustment as `boxes` is typically a collection, not a count)

5. **Output**:
   ```python
   print(boxes)
   ```
   - Prints information about detected objects' bounding boxes

## Requirements

- Python 3.x
- Ultralytics YOLO package (`pip install ultralytics`)
- OpenCV (installed automatically with ultralytics)
- Webcam connected to your system

## Usage

1. Install the required packages
2. Place the YOLO model file ('yolo11n.pt') in your working directory
3. Run the script: `python webcam.py`
4. View the real-time object detection in the displayed window
5. Press 'q' to quit the video display

## Customization Options

- Replace 'yolo11n.pt' with other YOLOv8 model variants for different speed/accuracy tradeoffs
- Adjust the `show` parameter to False if you don't want video display
- Modify the termination condition for different use cases
- Add additional processing for the detected objects (tracking, counting, etc.)

## Note

The script currently has a minor logical issue where it compares `boxes` (a collection) with a number. To count detected objects, you might want to use `len(boxes)` instead.
