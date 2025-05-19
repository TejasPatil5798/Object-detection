import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Use a recorded video file instead of webcam
video_path = r'C:\Users\tejas\Downloads\demo_vid.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for output
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

print("Processing video...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Done processing video.")
            break

        # Perform object detection
        results = model(frame)

        # Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = f'{class_id} {confidence:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the frame to output video
        out.write(frame)

        # Optional: Show progress frame-by-frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

# Release resources
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print(f"OpenCV GUI error: {e}")
