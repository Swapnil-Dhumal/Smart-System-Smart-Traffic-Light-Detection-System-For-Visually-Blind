import cv2
from ultralytics import YOLO

# Define the path to the model
model_path = ".\\yolov8_best.pt"
model = YOLO(model_path)

def get_predicted_class_and_boxes(image):
    try:
        # Perform object detection on the image
        results = model(image)

        # Check if any detections are made
        if len(results[0].boxes) > 0:
            # Extract the class IDs, confidence scores, and boxes
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy()  # Class IDs
            confidences = boxes.conf.cpu().numpy()  # Confidence scores
            box_coords = boxes.xyxy.cpu().numpy()  # Box coordinates (x1, y1, x2, y2)

            # Find the index of the highest confidence score
            max_conf_idx = confidences.argmax()

            # Get the corresponding class name
            highest_confidence_class = results[0].names[int(class_ids[max_conf_idx])]

            return highest_confidence_class.capitalize(), box_coords, class_ids, confidences, results[0].names
        return "Unknown", [], [], [], []  # No detections
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown", [], [], [], []

# Load the video
video_path = r"C:\Users\anura\OneDrive\Desktop\WhatsApp Video 2024-11-28 at 19.40.43_3c1be3ae.mp4"
cap = cv2.VideoCapture(video_path)

# Get the frame width, height, and FPS from the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output video path and codec for saving the video
output_video_path = r"C:\Users\anura\OneDrive\Desktop\output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # If the video ends, exit the loop

    # Get the predicted class, boxes, and other details for the current frame
    predicted_class, box_coords, class_ids, confidences, class_names = get_predicted_class_and_boxes(frame)

    # Display the class name on the frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw bounding boxes and labels
    for i, box in enumerate(box_coords):
        x1, y1, x2, y2 = box
        label = f"{class_names[int(class_ids[i])]}: {confidences[i]:.2f}"

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Bounding box
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Label

    # Show the frame with bounding boxes and predictions
    cv2.imshow("Video with YOLO Detection", frame)

    # Write the frame to the output video
    out.write(frame)

    # Wait for key press to continue or stop (e.g., press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and output writer, then close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
