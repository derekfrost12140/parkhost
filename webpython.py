from flask import Flask, Response, render_template, current_app
from queue import Queue
import cv2
from ultralytics import YOLO
from threading import Thread

#video path and other configs
#VIDEO_PATH = "/Users/ericgotcher/projects/Python/ParkingLot/videos/istockphoto-1581630757-640_adpp_is.mp4" #good
#VIDEO_PATH = "/Users/ericgotcher/projects/Python/ParkingLot/videos/istockphoto-1370353417-640_adpp_is.mp4"
VIDEO_PATH = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/istockphoto-1046782266-640-adpp-is_dNpvycW4.mp4?alt=media&token=52ae3e31-e5a5-4beb-8577-2d77f2f71474" #best
MODEL_PATH = "https://firebasestorage.googleapis.com/v0/b/parking-with-devops.appspot.com/o/best.pt?alt=media&token=c8a71ce0-7731-400d-93eb-b4b36ce8c709"
classNames = ["Empty", "Space Taken"]

# Initialize Flask app and queue
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
queue = Queue()

# Video processing function
def processVideo():
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        model = YOLO(MODEL_PATH)

        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            processedFrame = process_frame(frame, results, classNames)
            queue.put(processedFrame)

        cap.release()
    except Exception as e:
        print(f"An error occurred: {e}")


# Frame processing functions
def process_frame(frame, results, classNames):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            draw_box(frame, box, classNames)
    return frame

def draw_box(frame, box, classNames):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0,0,255) if classNames[cls] == 'Space Taken' else (0,255,0)
    if box.conf[0] > 0.43 and x2-x1 < 100:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        #cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

# Frame generation function
def generate():
    while True:
        frame = queue.get()
        ret, buffer = cv2.imencode('.jpg', frame)
        frameData = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frameData + b'\r\n')
        queue.task_done()

# Flask routes
"""@app.route("/")
@app.route("/video_feed")
def index():
    heading = "Lot X"
    title = "DEVVOPS Parking"
    video_source_url = "/video_feed"  # URL to the video feed
    return render_template('index.html', heading=heading, title=title, video_source_url=video_source_url)"""

# Flask route for video feed
@app.route("/")
def video_feed():
    heading = "Lot X"
    title = "DEVVOPS Parking"
    video_source_url = "/video_feed"  # URL to the video feed
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Main thread
if __name__ == "__main__":
    process_thread = Thread(target=processVideo)
    process_thread.daemon = True
    process_thread.start()

    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)