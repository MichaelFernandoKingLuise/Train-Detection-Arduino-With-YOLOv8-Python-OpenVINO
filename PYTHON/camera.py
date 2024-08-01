import time
import cv2
from ultralytics import YOLO

def cameraon():
# Initialize YOLO model
    model = YOLO("toy-train-det-01.pt", task="detect")

# Open the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Main loop for camera detection
    while cap.isOpened():
    # Read a frame from the video
        success, frame = cap.read()

        if success:
        # Run YOLOv8 inference on the frame
            results = model(frame, max_det=1)
            if len(results[0].boxes.cls) != 0:
                # [0] == KERETA API
                if results[0].boxes.cls[0] == 0:

        # Visualize the results on the frame
                    annotated_frame = results[0].plot()

        # Display the annotated frame
                    cv2.imshow("YOLOv8 Inference", annotated_frame)

# Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
