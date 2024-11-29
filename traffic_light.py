from ultralytics import YOLO

# Define the path to the model
model_path = ".\\yolov8_best.pt"
model = YOLO(model_path)

def get_predicted_class(image_path):
    try:
        # Perform object detection on the image
        results = model(image_path)

        # Check if any detections are made
        if len(results[0].boxes) > 0:
            # Extract the class IDs and confidence scores
            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy()  # Class IDs
            confidences = boxes.conf.cpu().numpy()  # Confidence scores

            # Print all detections with their confidence scores
            print("Detected classes and their confidence scores:")
            for cls_id, conf in zip(class_ids, confidences):
                class_name = results[0].names[int(cls_id)]
                # print(f"Class: {class_name}, Confidence: {conf:.2f}")

            # Find the index of the highest confidence score
            max_conf_idx = confidences.argmax()

            # Get the corresponding class name
            highest_confidence_class = results[0].names[int(class_ids[max_conf_idx])]
            return highest_confidence_class.capitalize()

        return "Unknown"  # No detections
    except Exception as e:
        print(f"Error: {e}")
        return "Unknown"

# # Example usage
# image_path = r"C:\Users\anura\OneDrive\Desktop\Colourblind_traffic_signal.webp"
# print(f"Predicted Class: {get_predicted_class(image_path)}")
