from flask import Flask, Response, render_template, request
import cv2
from ultralytics import YOLO
from queue import Queue
from threading import Thread
import logging
import os
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Setting a maxsize for the queue to prevent excessive memory usage
queue = Queue(maxsize=10)
# Configuration variables and global variables
DEFAULT_VIDEO_PATH = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1046782266-640-adpp-is_dNpvycW4.mp4?alt=media&token=52ae3e31-e5a5-4beb-8577-2d77f2f71474"
# Get the current directory of the script
current_directory = os.path.dirname(os.path.realpath(__file__))
# Set the MODEL_PATH to the correct location of best.pt
MODEL_PATH = os.path.join(current_directory, "best.pt")
classNames = ["Empty", "Space Taken"]
# Lazy model loading
model = None
def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model
# Initialize process_thread
process_thread = None
# Setup logging
logging.basicConfig(level=logging.INFO)
# Video processing function
def process_video(selected_video_path):
    global process_thread
    cap = cv2.VideoCapture(selected_video_path)
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return
    current_model = get_model()  # Load model on first use
    frame_skip = 5  # Number of frames to skip
    frame_count = 0
    try:
        while True:
            if process_thread and process_thread.stopped():
                break
            success, frame = cap.read()
            if not success or frame_count % frame_skip != 0:
                frame_count += 1
                continue
            # Reducing frame resolution
            frame = cv2.resize(frame, (640, 480))
            results = current_model(frame)
            processed_frame = process_frame(frame, results, classNames)
            if not queue.full():
                queue.put(processed_frame)
            frame_count += 1
    finally:
        cap.release()
        logging.info("Video capture released")
# Frame processing function
def process_frame(frame, results, classNames):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            draw_box(frame, box, classNames)
    return frame
def draw_box(frame, box, classNames):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0, 255, 0) if classNames[cls] == "Empty" else (0, 0, 255)
    if box.conf[0] > 0.43 and x2-x1 < 100:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
# Frame generation for Flask
def generate():
    while True:
        frame = queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frameData = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frameData + b'\r\n')
        queue.task_done()
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/")
def parking():
     return render_template('lots.html')
@app.route("/show")
def show():
    return render_template('index.html')
class StoppableThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = False
    def stop(self):
        self._stop_event = True
    def stopped(self):
        return self._stop_event
@app.route("/select_lot", methods=["POST"])
def select_lot():
    lot_number = request.form.get('lot')
    # Hardcoded URLs
    video_paths = {
        "1": "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/gettyimages-1533928757-640_adpp.mp4?alt=media&token=d2fb383f-8504-4d46-a977-2b1e02a6fbfa",
        "2": "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1046782266-640-adpp-is_dNpvycW4.mp4?alt=media&token=52ae3e31-e5a5-4beb-8577-2d77f2f71474",
        "3": "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1370353417-640_adpp_is.mp4?alt=media&token=d4cd845b-8b18-4bcf-8be1-b1bc90d26138",
        "4": "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-845199510-640_adpp_is.mp4?alt=media&token=fdbbd0dc-db3a-46e4-b6d3-afa2583cdfba"
    }
    selected_video_path = video_paths.get(lot_number)
    if not selected_video_path:
        return "Invalid lot number", 400
    logging.info(f"Selected lot: {lot_number}, video: {selected_video_path}")
    global process_thread
    if process_thread and process_thread.is_alive():
        process_thread.stop()
        process_thread.join()
    process_thread = StoppableThread(target=process_video, args=(selected_video_path,))
    process_thread.daemon = True
    process_thread.start()
    return "Video selected successfully"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
