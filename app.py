from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
from PIL import Image
from io import BytesIO
import os
from ultralytics import YOLO
from traffic_light import get_predicted_class
import traffic_light

app = Flask(__name__)
CORS(app)    
 
SAVE_DIR = "static"
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(".\\best.pt")  # Replace with the path to your trained pothole detection model
class_names = model.names

#/detect_light
#/detect_pothole

import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS


def speak_marathi(text):
    tts = gTTS(text=text, lang='mr')
    filename = os.path.join(".\\static","message.mp3" )
    print("New File Saved : ", text)
    tts.save(filename)  


def detect_traffic_light_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    predicted_class = get_predicted_class(hsv_image) 
    print("Predicted Class Is : " , predicted_class)
    return predicted_class

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/home")
def homepage():
    return render_template("index.html")


@app.route("/audio-file")
def audio_return():
    return send_from_directory('static', 'message.mp3', as_attachment=True)


@app.route("/audio-beep")
def beep_return():
    return send_from_directory('static', 'beep.mp3', as_attachment=True)

# iput imgae get
@app.route("/detect_light", methods=["POST"])
def upload_light():
    try: 
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
         
        image_data = data["image"].split(",")[1]
        decoded_image = base64.b64decode(image_data) 
         
        image = Image.open(BytesIO(decoded_image)) 
        
        image_filename = os.path.join(SAVE_DIR, "frame.png")
        image.save(image_filename)

        color = detect_traffic_light_color(np.array(image))

        marathi_message = {
                            "Red": "वेट करा, सिग्नल हिरवा होईपर्यंत.",
                            "Green": "जा, सिग्नल हिरवा आहे.",
                            "Yellow": "तुम्ही जाऊ शकता, पण काळजी घ्या.",
                            "Unknown": "सिग्नल ओळखू शकत नाही."
                        }

        speak_marathi(marathi_message.get(color.strip() , "सिग्नल ओळखू शकत नाही.") )
        return jsonify({ "file_path": image_filename , "color" : color , "message" : marathi_message.get(color , "सिग्नल ओळखू शकत नाही.")  }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500
    


@app.route("/detect_pothole", methods=["POST"])
def upload_pothole():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
         
        image_data = data["image"].split(",")[1]
        decoded_image = base64.b64decode(image_data) 
         
        image = Image.open(BytesIO(decoded_image)) 
        img = np.array(image)
        img = cv2.cvtColor(img , cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (1020, 500)) 
        h, w, _ = img.shape
        results = model.predict(img) 
        # print(results)
        print(results[0].boxes)
        # print(results) 


        if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
            message = "Pothole Detected"
        else:
            message = "No Potholes" 
        pothole_detected = False

        
        image_filename = os.path.join(SAVE_DIR, "frame.png")
        image.save(image_filename)
        
        return jsonify({  "file_path": image_filename, "pothole":message }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process image", "details": str(e)}), 500



def get_predicted_class_and_boxes(image):
    try:
        # Perform object detection on the image
        results = traffic_light.model(image)

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


def add_overlay_to_frame(frame):
    """Add an overlay (e.g., text or shapes) to the frame."""
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

    marathi_message = {
                            "Red": "वेट करा, सिग्नल हिरवा होईपर्यंत.",
                            "Green": "जा, सिग्नल हिरवा आहे.",
                            "Yellow": "तुम्ही जाऊ शकता, पण काळजी घ्या." 
                        }
    
    speak_marathi(marathi_message.get(predicted_class, "सिग्नल ओळखू शकत नाही."))

    return frame,predicted_class


def get_predicted_class_and_boxes2(image):
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

def add_overlay_to_frame2(frame):
    """Add an overlay (e.g., text or shapes) to the frame."""
    predicted_class, box_coords, class_ids, confidences, class_names = get_predicted_class_and_boxes2(frame)

    # Display the class name on the frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw bounding boxes and labels
    for i, box in enumerate(box_coords):
        x1, y1, x2, y2 = box
        label = f"{class_names[int(class_ids[i])]}: {confidences[i]:.2f}"

        # Draw the bounding box and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Bounding box
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
    if(predicted_class):
        class_detected = "Pothole"
    else:
        class_detected = "None"
        
    return frame,class_detected


# video traffic
@app.route('/process_frame', methods=['POST'])    
def process_frame():
    data = request.json
    if 'frame' not in data:
        return jsonify({'error': 'No frame data received'}), 400

    # Decode base64 image
    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    np_frame = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Process the frame
    processed_frame , predicted_class = add_overlay_to_frame(frame)

    # Encode processed frame to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'processed_frame': f'data:image/jpeg;base64,{processed_frame_b64}' , "class": predicted_class})


# pothole
@app.route('/process_frame_potholes', methods=['POST'])
def process_frame_pothole():
    data = request.json
    if 'frame' not in data:
        return jsonify({'error': 'No frame data received'}), 400

    # Decode base64 image
    frame_data = data['frame'].split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    np_frame = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # Process the frame
    processed_frame , predicted_class = add_overlay_to_frame2(frame)

    # Encode processed frame to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'processed_frame': f'data:image/jpeg;base64,{processed_frame_b64}' , "class": predicted_class})


@app.route('/video_traffic'  )
def video_traffic():
    return render_template("video_traffic.html")

@app.route('/video_pothole'  )
def video_pothole():
    return render_template("video_potholes.html")

@app.route('/image_process'  )
def image_process():
    return render_template("image_process.html")


# @app.route('/process-traffic-light'  )
# def image_process_traffic():
#     pass

# @app.route('/process-potholes'  )
# def image_process_potholes():
#     pass


if __name__ == "__main__":
    app.run(debug=True, port=5000)
