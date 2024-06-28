import cv2
import firebase_admin
from firebase_admin import credentials, storage, db
import datetime
import time
import os
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import logging
import numpy as np
import urllib.request

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase Admin SDK
cred = credentials.Certificate('graduation-project-23499-firebase-adminsdk-hek9v-6b72aee967.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'graduation-project-23499.appspot.com',
    'databaseURL': 'https://graduation-project-23499-default-rtdb.europe-west1.firebasedatabase.app/'
})
bucket = storage.bucket()

ESP32_CAM_URL = "http://192.168.1.132/"
FRAME_RATE = 2  # Set consistent frame rate for capturing frames

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes for YOLOv4-tiny
classes = ["person", "animal", "fire", "smoke"]  # Adjust as per your needs

def save_frame_locally(frame, frame_count, date_str):
    try:
        os.makedirs(f'frames/{date_str}', exist_ok=True)
        frame_path = os.path.abspath(f'frames/{date_str}/frame_{frame_count:06d}.jpg')
        cv2.imwrite(frame_path, frame)
        logging.info(f"Frame {frame_count} saved locally at {frame_path}")
        return frame_path
    except Exception as e:
        logging.error(f"Failed to save frame {frame_count} locally: {e}")
        return None

def add_datetime_text(frame):
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(frame, date_time_str,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    return frame

def create_video_from_frames(date_str, part_number):
    frame_folder = f'frames/{date_str}'
    video_path = f'videos/{date_str}_part{part_number}.mp4'
    os.makedirs('videos', exist_ok=True)

    frame_files = sorted([os.path.abspath(os.path.join(frame_folder, f)) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
    if not frame_files:
        logging.info(f"No frames found for {date_str}")
        return None

    frame_list_file = os.path.abspath(f'{frame_folder}/frame_list_{part_number}.txt')
    with open(frame_list_file, 'w') as f:
        for frame_file in frame_files:
            f.write(f"file '{frame_file}'\n")
    logging.info(f"Frame list created at {frame_list_file}")

    cmd = [
        'ffmpeg', '-y', '-r', str(FRAME_RATE), '-f', 'concat', '-safe', '0', '-i', frame_list_file,
        '-vsync', 'vfr', '-pix_fmt', 'yuv420p', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    logging.info(result.stdout)
    logging.error(result.stderr)

    if os.path.exists(video_path):
        logging.info(f"Video created for {date_str} at {video_path}")

        # Clean up frames and frame list after creating video
        for frame_file in frame_files:
            os.remove(frame_file)
        os.remove(frame_list_file)
        logging.info(f"Frames and frame list deleted for {date_str}, part {part_number}")

        return video_path
    else:
        logging.error(f"Failed to create video for {date_str}")
        os.remove(frame_list_file)
        return None

def upload_to_firebase(local_path, remote_path):
    try:
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        logging.info(f"{local_path} uploaded to Firebase Storage at {remote_path}")
        os.remove(local_path)  # Remove file after uploading
        logging.info(f"{local_path} deleted after upload")
    except Exception as e:
        logging.error(f"Failed to upload {local_path}: {e}")

def fetch_frame_from_esp32():
    try:
        with urllib.request.urlopen(ESP32_CAM_URL) as response:
            image_data = response.read()
        image_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logging.error(f"Failed to fetch frame from ESP32: {e}")
        return None

def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_objects.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, detected_objects

def log_event(event_type, date_str, part_number, stream_name, log_file):
    now = datetime.datetime.now()
    event_time = now.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"Event: {event_type}, Time: {event_time}, Date: {date_str}, Part: {part_number}, Stream: {stream_name}\n"
    logging.info(log_message)
    with open(log_file, 'a') as f:
        f.write(log_message)

def capture_and_process_frames():
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    frame_count = 0
    last_video_creation_time = time.time()
    part_number = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        while True:
            start_time = time.time()
            frame = fetch_frame_from_esp32()
            if frame is None:
                continue

            log_file = f'logs/log_part{part_number}.txt'

            if frame_count % 10 == 0:
                frame, detected_objects = detect_objects(frame)
                if detected_objects:
                    for obj in detected_objects:
                        log_event(obj, current_date_str, part_number, 'stream1', log_file)

            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            if date_str != current_date_str:
                current_date_str = date_str
                frame_count = 0
                part_number = 0

            frame = add_datetime_text(frame)
            frame_path = save_frame_locally(frame, frame_count, current_date_str)
            frame_count += 1

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_base64 = base64.b64encode(frame).decode('utf-8')
            try:
                db.reference('streams/stream1').set(frame_base64)
                logging.info(f"Frame {frame_count} pushed to Firebase Realtime Database")
            except Exception as e:
                logging.error(f"Error updating Firebase Realtime Database: {e}")

            if isinstance(frame, np.ndarray):
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                logging.error(f"Invalid frame type: {type(frame)}")

            if time.time() - last_video_creation_time >= 60:
                last_video_creation_time = time.time()
                future = executor.submit(handle_video_creation_and_upload, current_date_str, part_number)
                futures.append(future)
                part_number += 1

            time.sleep(max(0, (1 / FRAME_RATE) - (time.time() - start_time)))  # Ensure consistent frame rate

        for future in as_completed(futures):
            future.result()

    cv2.destroyAllWindows()

def handle_video_creation_and_upload(date_str, part_number):
    video_path = create_video_from_frames(date_str, part_number)
    if video_path:
        upload_to_firebase(video_path, f'camera1/{date_str}/{date_str}_part{part_number}.mp4')
    log_file = f'logs/log_part{part_number}.txt'
    if os.path.exists(log_file):
        upload_to_firebase(log_file, f'camera1_logs/{date_str}_log_part{part_number}.txt')

if __name__ == '__main__':
    logging.info("Starting the stream to Firebase...")
    capture_and_process_frames()
