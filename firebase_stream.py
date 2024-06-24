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
import urllib.request  # Add this import

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Firebase Admin SDK
cred = credentials.Certificate('graduation-project-23499-firebase-adminsdk-hek9v-6b72aee967.json')  # Replace with the correct path
firebase_admin.initialize_app(cred, {
    'storageBucket': 'graduation-project-23499.appspot.com',  # Replace with your storage bucket URL
    'databaseURL': 'https://graduation-project-23499-default-rtdb.europe-west1.firebasedatabase.app/'  # Replace with your database URL
})
bucket = storage.bucket()

ESP32_CAM_URL = "http://192.168.1.208/"

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
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', frame_list_file,
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

def upload_video_to_firebase(video_path, date_str, part_number):
    try:
        blob = bucket.blob(f'camera2/{date_str}_part{part_number}.mp4')
        blob.upload_from_filename(video_path)
        logging.info(f"Video {video_path} uploaded to Firebase Storage")
        os.remove(video_path)  # Remove video file after uploading
        logging.info(f"Video file {video_path} deleted after upload")
    except Exception as e:
        logging.error(f"Failed to upload video {video_path}: {e}")

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

def capture_and_process_frames():
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    frame_count = 0
    last_video_creation_time = time.time()
    frame_rate = 10  # Frame rate for video creation
    part_number = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        while True:
            frame = fetch_frame_from_esp32()
            time.sleep(4)
            if frame is None:
                continue

            # Get current date
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            if date_str != current_date_str:
                current_date_str = date_str
                frame_count = 0
                part_number = 0

            # Add date and time text to the frame
            frame = add_datetime_text(frame)

            # Save frame locally
            frame_path = save_frame_locally(frame, frame_count, current_date_str)
            frame_count += 1

            # Encode frame to base64 and push to Firebase Realtime Database
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            frame_base64 = base64.b64encode(frame).decode('utf-8')
            try:
                db.reference('camera/stream1').set(frame_base64)
                logging.info(f"Frame {frame_count} pushed to Firebase Realtime Database")
            except Exception as e:
                logging.error(f"Error updating Firebase Realtime Database: {e}")

            # Check if it's time to create and upload a video
            if time.time() - last_video_creation_time >= 60:  # Every 1 minute
                last_video_creation_time = time.time()
                future = executor.submit(handle_video_creation_and_upload, current_date_str, frame_rate, part_number)
                futures.append(future)
                part_number += 1

        for future in as_completed(futures):
            future.result()

def handle_video_creation_and_upload(date_str, frame_rate, part_number):
    video_path = create_video_from_frames(date_str, part_number)
    if video_path:
        upload_video_to_firebase(video_path, date_str, part_number)

if _name_ == '_main_':
    logging.info("Starting the stream to Firebase...")
    capture_and_process_frames()
