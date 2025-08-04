import asyncio
import gc
import logging
import multiprocessing
import os
import platform
import queue
import subprocess
import time
import threading
from urllib.parse import urlparse
import cv2
from fastapi import Request
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import torch
from concurrent.futures import ThreadPoolExecutor
from camera import FreshestFrame
from savatoDb import load_embeddings_from_db, insertToDb

# --- Basic Setup ---
logging.getLogger('torch').setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
frps = 5 if device == 'cuda' else 25


cv2.setNumThreads(multiprocessing.cpu_count())
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
# Frame rate limiter (FPS)
TARGET_FPS = 30  # Adjust based on your needs
FRAME_DELAY = 1.0 / TARGET_FPS
# Constants for health monitoring
RETRY_LIMIT = 5
RETRY_DELAY = 3  # seconds
model = None
face_handler = None
known_names = {}
process = None
# Load known embeddings


lock = threading.Lock()
recognition_queue = queue.Queue()
# {track_id: {'name': str, 'bbox': (x1,y1,x2,y2), 'last_update': time.time()}}
face_info = {}
face_info_lock = threading.Lock()
embedding_cache = {}  # Optional cache {track_id: embedding}


executor = ThreadPoolExecutor(max_workers=4)  # Recognition threads


# --- Functions ---

def realseFreshest(fresh: FreshestFrame, cap: cv2.VideoCapture):
    try:
        fresh.release()
        cap.release()
        recognition_queue.put(None)

    except Exception as e:
        logging.error(f"Error to Realse Cameras : {e}")


def loadModel():
    global model, face_handler
    if model == None:
        face_handler = FaceAnalysis('antelopev2', providers=[
            'CUDAExecutionProvider', 'CPUExecutionProvider'],root='.')
        face_handler.prepare(ctx_id=0)
        model = YOLO(MODEL_PATH, verbose=False)
        model.eval()
        logging.info('Model Load.')
    with lock:
        return True


def loadConfig():
    pass


def loadDb():

    global known_names, process
    try:
        # process = subprocess.Popen(
        #     ["pocketbase", "serve", "--http=0.0.0.0:8090"], creationflags=subprocess.CREATE_NO_WINDOW,)
        # logging.info(f"PocketBase stater {process.pid}")
        known_names = load_embeddings_from_db()

    except Exception as e:
        logging.error(e)
        known_names = {}


def graceful_shutdown():

    # Clean up resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Stop observer if running

    logging.info("Cleanup complete. Shutting down.")
    process.terminate()

    os._exit(0)  # Use os._exit instead of sys.exit for more forceful termination


def update_face_info(track_id, name, score, gender, age, role, bbox=None):
    with face_info_lock:
        face_info[track_id] = {'name': name, 'bbox': bbox,
                               'last_update': time.time(), 'score': score, 'gender': gender, 'age': age, 'role': role}


def recognize_face(embedding, fgender, fage):
    best_match = 'unknown'
    best_score = 0.0
    best_age = fage  # Default to detected age
    best_gender = fgender  # Default to detected gender
    best_role = ''
    for name, person_data in known_names.items():
        age = person_data['age']
        gender = person_data['gender']
        role = person_data['role']
        embeds = person_data['embeddings']
        for known_emb in embeds:
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > best_score:
                best_score = sim
                best_match = name
                best_age = age  # Store the known age
                best_gender = gender  # Store the known gender
                best_role = role
    if best_score >= 0.6:

        return best_match, best_score, best_gender, best_age, best_role
    else:
        return "unknown", best_score, fgender, fage, best_role


def recognition_worker():

    logging.info("Recognition thread started.")
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        track_id, face_img = item

        # Optional caching: skip if recently updated
        if track_id in face_info and time.time() - face_info[track_id]['last_update'] < 2:
            continue

        faces = face_handler.get(face_img)

        if faces:
            face = faces[0]
            gender = 'female' if face.gender == 0 else 'male'
            age = face.age

            name, sim, gender, age, role = recognize_face(
                face.embedding, gender, age)
            x1, y1, x2, y2 = map(int, face.bbox)

            update_face_info(track_id, name, sim, gender,
                             age, role, (x1, y1, x2, y2))
            embedding_cache[track_id] = face.embedding
        else:
            update_face_info(track_id, "Unknown", 'None', 'None', '', None,)


# --- Threads ---

threading.Thread(target=recognition_worker, daemon=True).start()


# --- Main Loop ---

async def process_frame(frame, path, counter):
    try:

        start_time = time.time()

        while True:
            if frame.size == 0:
                continue

            frame = cv2.resize(frame, (640, 640))

            results = model.track(
                frame, classes=[0], tracker="bytetrack.yaml", persist=True, device=device)

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4].cpu().tolist())
                    try:
                        track_id = int(box.id[0].cpu().tolist())
                    except Exception:
                        continue

                    human_crop = frame[y1:y2, x1:x2]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Only recognize every frps frames
                    if counter % frps == 0:
                        recognition_queue.put((track_id, human_crop))

                    info = face_info.get(
                        track_id, {'name': "Unknown", "score": 0, 'bbox': None, 'gender': 'None', "age": "None"})
                    label = f"{info['name']} ID:{track_id}"
                    face_bbox = info['bbox']

                    try:
                        score = int(info['score']*100)
                    except TypeError:
                        score = 0
                    name = info['name']
                    gender = info['gender']
                    age = info['age']
                    role = info['role']
                    if face_bbox:
                        fx1, fy1, fx2, fy2 = face_bbox
                        cv2.rectangle(frame, (x1 + fx1, y1 + fy1),
                                      (x1 + fx2, y1 + fy2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        heightf, widthf = human_crop.shape[:2]
                        padding = 40
                        fx1 = max(fx1 - padding, 0)
                        fy1 = max(fy1 - padding, 0)
                        fx2 = min(fx2 + padding, widthf)
                        fy2 = min(fy2 + padding, heightf)
                        # TODO:FIX THIS , AND lest get ot ui
                        croppedface = human_crop[fy1:fy2, fx1:fx2]

                        try:
                            await insertToDb(name, frame, croppedface, human_crop,  # 3 pic (croppedface,croppedhuman,frame) send role to db
                                             score, track_id, gender, age, role, path)
                        except Exception as e:
                            logging.error(f"Error Insert to DB {e}")
                            continue

                    else:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FPS calc
            fps = 1.0 / (time.time() - start_time)
            start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return frame
    except Exception as e:
        return frame


def isConnectionAlive(source):
    ulr = urlparse(source).hostname
    param = "-n" if platform.system().lower() == "windows" else "-c"

    # Build the ping command
    command = ["ping", param, "1", ulr]

    try:
        # Execute the ping command
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


async def generate_frames(camera_idx, source, request: Request):
    """Generate frames from a specific camera feed"""
    # loadModel()
    if not isConnectionAlive(source):
        return
    """Generate frames from a specific camera feed"""

    check_interval = 60  # seconds
    last_check = 0

    def open_capture(source):
        cap = cv2.VideoCapture(source)
        return cap if cap.isOpened() else None

    retries = 0
    cap = open_capture(source)

    while cap is None and retries < RETRY_LIMIT:
        logging.error(
            f"[Camera {camera_idx}] Failed to open source. Retrying ({retries + 1}/{RETRY_LIMIT})...")
        await asyncio.sleep(RETRY_DELAY)
        retries += 1
        cap = open_capture(source)

    if cap is None:
        logging.error(
            f"[Camera {camera_idx}] Could not open source after {RETRY_LIMIT} retries.")
        return
    fresh = FreshestFrame(cap)

    try:
        while fresh.is_alive():
            now = time.time()
            if now - last_check >= check_interval:

                if not isConnectionAlive(source):
                    break
                last_check = now

            if await request.is_disconnected():
                logging.info("Client disconnected, releasing camera.")
                break
            success, frame = fresh.read()
            height, width = frame.shape[0], frame.shape[1]
            if not success:

                # Generate blank frame if we can't read from camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No signal", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:

                frame = await process_frame(frame, f'/rt{camera_idx}', success)

            frame = cv2.resize(frame, (width, height))

            # Encode and yield the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        realseFreshest(fresh, cap)


def imageSearcher(filePath):
    frame = cv2.imread(filePath)
    _, img_encoded = cv2.imencode(".jpg", frame)
    return img_encoded


def imageCrop(filepath):
    frame = cv2.imread(filepath)
    facebox = face_handler.get(frame)[0].bbox
    x1, y1, x2, y2 = map(int, facebox)
    heightf, widthf = frame.shape[:2]
    padding = 40
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, widthf)
    y2 = min(y2 + padding, heightf)
    frame = frame[y1:y2, x1:x2]
    _, img_encoded = cv2.imencode(".jpg", frame)
    return img_encoded


if __name__ == "__main__":
    imageCrop(r'dbimage\aref\image.png')
