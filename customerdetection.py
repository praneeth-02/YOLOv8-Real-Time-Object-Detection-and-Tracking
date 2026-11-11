
import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check webcam availability
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Customer Detection Started...")
print("Press 'q' to exit.")

while True:

    # Capture video frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read video frame.")
        break

    # Perform object detection
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    # Get detected objects
    annotated_frame = results[0].plot()

    # Display detection window
    cv2.imshow(
        "YOLOv8 Real-Time Customer Detection",
        annotated_frame
    )

    # Press q to stop detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()

print("Customer Detection Stopped.")
