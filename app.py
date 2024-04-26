from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import os
import cv2
import uuid
import tempfile
import logging
from ultralytics import YOLO

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Specify your Netlify domain in the CORS settings
cors_origins = ["https://main--bespoke-concha-8d1e60.netlify.app"]

# Enable CORS for all routes and specify origins to allow
CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=True)

# Load YOLO model
model_url = os.getenv("MODEL_URL", "default_model_url_here")
model = YOLO(model_url)
classNames = ["Empty", "Space Taken"]

@app.route('/')
def index():
    return send_from_directory('templates', 'lot.html')

@app.route('/process', methods=['POST'])
@cross_origin()  # Apply CORS for this endpoint
def process_video():
    try:
        video_url = request.json['url']
        cap = cv2.VideoCapture(video_url)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_filename = f"{uuid.uuid4()}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            success, img = cap.read()
            if not success:
                break
            results = model(img, stream=True)
            processed_frame = process_frame(img, results)
            out.write(processed_frame)
        
        cap.release()
        out.release()
        return jsonify({'message': 'Processing complete', 'video': f'/videos/{output_filename}'})
    except Exception as e:
        logging.error("Failed to process video: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(tempfile.gettempdir(), filename)

def process_frame(img, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            draw_box(img, box)
    return img

def draw_box(img, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{classNames[cls]}: {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
