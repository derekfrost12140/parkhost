import os
import cv2
from ultralytics import YOLO
import logging
from redis import Redis
from rq import Worker, Queue, Connection

# Set up logging
logging.basicConfig(level=logging.INFO)

# Redis connection setup for RQ
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_conn = Redis.from_url(redis_url)

# Configuration variables
MODEL_PATH = os.path.join(os.getcwd(), "best.pt")
classNames = ["Empty", "Space Taken"]

# Lazy model loading
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

def process_video(selected_video_path):
    cap = cv2.VideoCapture(selected_video_path)
    if not cap.isOpened():
        logging.error("Error opening video stream or file")
        return
    current_model = get_model()
    frame_skip = 5
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success or frame_count % frame_skip != 0:
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 480))
            results = current_model(frame)
            processed_frame = process_frame(frame, results)

            # Here, you could either store the processed frame or handle it in some way
            logging.info("Processed a frame")
            frame_count += 1

    finally:
        cap.release()
        logging.info("Video capture released")

def process_frame(frame, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            draw_box(frame, box)
    return frame

def draw_box(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0, 255, 0) if classNames[cls] == "Empty" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

if __name__ == "__main__":
    # Listen to the default queue
    with Connection(redis_conn):
        worker = Worker(map(Queue, ['default']))
        worker.work()

