import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the appropriate model file

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

print("Camera opened successfully. Capturing image...")

# Read a single frame from the camera
ret, frame = cap.read()
if not ret:
    print("Error: Failed to read frame from camera")
else:
    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = f'{model.names[class_id]} {confidence:.2f}'  # Get class name from model
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Captured Image with Object Detection', frame)
    print("Press any key to exit...")
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

# Release resources
cap.release()
cv2.destroyAllWindows()
