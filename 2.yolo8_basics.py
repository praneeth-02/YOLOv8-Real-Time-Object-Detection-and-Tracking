from ultralytics import YOLO
import numpy

model = YOLO('yolov8n.pt',"v8")

detections = model.predict(source="https://ultralytics.com/images/bus.jpg", show=True)

print(detection_output)