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
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

cv2.setNumThreads(multiprocessing.cpu_count())

class CCtvMonitor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frps = 5 if self.device == 'cuda' else 25
        self.MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
        self.TARGET_FPS = 30
        self.FRAME_DELAY = 1.0 / self.TARGET_FPS
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        
        # Initialize models
        self.model = None
        self.face_handler = None
        self._load_models()
        
        # Load database
        self.known_names = self.load_db()
        
        # Threading and process management
        self.process = None
        self.lock = threading.Lock()
        self.recognition_queue = queue.Queue()
        self.face_info = {}
        self.face_info_lock = threading.Lock()
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown_event = threading.Event()

    def _load_models(self):
        """Load YOLO and face recognition models"""
        try:
            logging.info("Loading models...")
            
            # Load face handler
            self.face_handler = FaceAnalysis(
                'antelopev2', 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
                root='.'
            )
            self.face_handler.prepare(ctx_id=0)
            
            # Load YOLO model
            self.model = YOLO(self.MODEL_PATH, verbose=False)
            self.model.eval()
            
            logging.info('Models loaded successfully.')
            
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def load_db(self):
        """Load known faces from database"""
        try:
            known_names = load_embeddings_from_db()
            logging.info(f"Loaded {len(known_names)} known faces from database")
            return known_names
        except Exception as e:
            logging.error(f"Failed to load database: {e}")
            return {}

    def release_resources(self, fresh: FreshestFrame, cap: cv2.VideoCapture):
        """Properly release camera resources"""
        try:
            if fresh:
                fresh.release()
            if cap:
                cap.release()
            # Signal recognition worker to stop
            self.recognition_queue.put(None)
        except Exception as e:
            logging.error(f"Error releasing camera resources: {e}")

    def graceful_shutdown(self):
        """Gracefully shutdown the system"""
        logging.info("Initiating graceful shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Garbage collection
        gc.collect()
        
        # Terminate subprocess if exists
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        logging.info("Cleanup complete.")

    def update_face_info(self, track_id, name, score, gender, age, role, bbox=None):
        """Thread-safe update of face information"""
        with self.face_info_lock:
            self.face_info[track_id] = {
                'name': name, 
                'bbox': bbox,
                'last_update': time.time(), 
                'score': score, 
                'gender': gender, 
                'age': age, 
                'role': role
            }

    def recognize_face(self, embedding, fgender, fage):
        """Recognize face using embedding comparison"""
        best_match = 'unknown'
        best_score = 0.0
        best_age = fage
        best_gender = fgender
        best_role = ''
        
        try:
            for name, person_data in self.known_names.items():
                age = person_data['age']
                gender = person_data['gender']
                role = person_data['role']
                embeds = person_data['embeddings']
                
                for known_emb in embeds:
                    sim = cosine_similarity([embedding], [known_emb])[0][0]
                    if sim > best_score:
                        best_score = sim
                        best_match = name
                        best_age = age
                        best_gender = gender
                        best_role = role
            
            # Threshold for recognition
            if best_score >= 0.6:
                return best_match, best_score, best_gender, best_age, best_role
            else:
                return "unknown", best_score, fgender, fage, best_role
                
        except Exception as e:
            logging.error(f"Error in face recognition: {e}")
            return "unknown", 0.0, fgender, fage, ""

    def recognition_worker(self):
        """Background worker for face recognition"""
        logging.info("Recognition worker started.")
        
        while not self._shutdown_event.is_set():
            try:
                # Use timeout to allow checking shutdown event
                item = self.recognition_queue.get(timeout=1.0)
                
                if item is None:
                    break
                    
                track_id, face_img = item
                
                # Skip if recently updated (performance optimization)
                with self.face_info_lock:
                    if (track_id in self.face_info and 
                        time.time() - self.face_info[track_id]['last_update'] < 2):
                        continue
                
                # Process face
                faces = self.face_handler.get(face_img)
                
                if faces:
                    face = faces[0]
                    gender = 'female' if face.gender == 0 else 'male'
                    age = face.age
                    
                    name, sim, gender, age, role = self.recognize_face(
                        face.embedding, gender, age
                    )
                    x1, y1, x2, y2 = map(int, face.bbox)
                    
                    self.update_face_info(
                        track_id, name, sim, gender, age, role, (x1, y1, x2, y2)
                    )
                    self.embedding_cache[track_id] = face.embedding
                else:
                    self.update_face_info(
                        track_id, "Unknown", 0.0, 'None', 'None', '', None
                    )
                    
            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                logging.error(f"Error in recognition worker: {e}")
                
        logging.info("Recognition worker stopped.")

    def start(self):
        """Start the recognition worker thread"""
        recognition_thread = threading.Thread(
            target=self.recognition_worker, 
            daemon=True
        )
        recognition_thread.start()
        return recognition_thread

    async def process_frame(self, frame, path, counter):
        """Process a single frame for object detection and face recognition"""
        try:
            if frame.size == 0:
                return frame

            start_time = time.time()
            
            # Resize frame for processing
            processed_frame = cv2.resize(frame, (640, 640))

            # Run YOLO detection
            results = self.model.track(
                processed_frame, 
                classes=[0],  # Person class
                tracker="bytetrack.yaml", 
                persist=True, 
                device=self.device
            )

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4].cpu().tolist())
                    
                    # Get tracking ID
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].cpu().item())

                    # Crop human region
                    human_crop = processed_frame[y1:y2, x1:x2]
                    if human_crop.size == 0:
                        continue

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Queue for recognition every frps frames
                    if counter % self.frps == 0:
                        self.recognition_queue.put((track_id, human_crop))

                    # Get face info
                    with self.face_info_lock:
                        info = self.face_info.get(
                            track_id, 
                            {
                                'name': "Unknown", 
                                'score': 0, 
                                'bbox': None, 
                                'gender': 'None', 
                                'age': 'None',
                                'role': ''
                            }
                        )

                    # Create label
                    label = f"{info['name']} ID:{track_id}"
                    face_bbox = info['bbox']

                    try:
                        score = int(info['score'] * 100) if info['score'] else 0
                    except (TypeError, ValueError):
                        score = 0

                    name = info['name']
                    gender = info['gender']
                    age = info['age']
                    role = info['role']

                    # Draw face bounding box if available
                    if face_bbox:
                        fx1, fy1, fx2, fy2 = face_bbox
                        cv2.rectangle(
                            processed_frame, 
                            (x1 + fx1, y1 + fy1),
                            (x1 + fx2, y1 + fy2), 
                            (0, 0, 255), 2
                        )
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )

                        # Crop face with padding
                        height_f, width_f = human_crop.shape[:2]
                        padding = 40
                        fx1_padded = max(fx1 - padding, 0)
                        fy1_padded = max(fy1 - padding, 0)
                        fx2_padded = min(fx2 + padding, width_f)
                        fy2_padded = min(fy2 + padding, height_f)
                        
                        cropped_face = human_crop[fy1_padded:fy2_padded, fx1_padded:fx2_padded]

                        # Insert to database
                        try:
                            await insertToDb(
                                name, processed_frame, cropped_face, human_crop,
                                score, track_id, gender, age, role, path
                            )
                        except Exception as e:
                            logging.error(f"Error inserting to DB: {e}")

                    else:
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(
                processed_frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            return processed_frame

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def is_connection_alive(self, source):
        """Check if network connection to source is alive"""
        try:
            url = urlparse(source).hostname
            if not url:
                return True  # Local source or invalid URL
                
            param = "-n" if platform.system().lower() == "windows" else "-c"
            command = ["ping", param, "1", url]
            
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
            
        except (subprocess.TimeoutExpired, Exception) as e:
            logging.warning(f"Connection check failed: {e}")
            return False

    async def generate_frames(self, camera_idx, source, request: Request):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        check_interval = 60  # seconds
        last_check = 0
        counter = 0

        def open_capture(source):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            return None

        # Retry logic for opening capture
        retries = 0
        cap = None
        
        while cap is None and retries < self.RETRY_LIMIT:
            cap = open_capture(source)
            if cap is None:
                logging.error(
                    f"[Camera {camera_idx}] Failed to open source. "
                    f"Retrying ({retries + 1}/{self.RETRY_LIMIT})..."
                )
                await asyncio.sleep(self.RETRY_DELAY)
                retries += 1

        if cap is None:
            logging.error(
                f"[Camera {camera_idx}] Could not open source after {self.RETRY_LIMIT} retries."
            )
            return

        fresh = FreshestFrame(cap)

        try:
            while fresh.is_alive() and not self._shutdown_event.is_set():
                now = time.time()
                
                # Periodic connection check
                if now - last_check >= check_interval:
                    if not self.is_connection_alive(source):
                        logging.warning(f"[Camera {camera_idx}] Connection lost")
                        break
                    last_check = now

                # Check if client disconnected
                if await request.is_disconnected():
                    logging.info("Client disconnected, releasing camera.")
                    break

                success, frame = fresh.read()
                counter += 1

                if not success or frame is None:
                    # Generate blank frame if we can't read from camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "No signal", (220, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                else:
                    # Store original dimensions
                    original_height, original_width = frame.shape[:2]
                    
                    # Process frame
                    frame = await self.process_frame(frame, f'/rt{camera_idx}', counter)
                    
                    # Resize back to original dimensions
                    frame = cv2.resize(frame, (original_width, original_height))

                # Encode and yield the frame
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logging.error(f"Error encoding frame: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
        finally:
            self.release_resources(fresh, cap)


def image_searcher(file_path):
    """Load and encode image for searching"""
    try:
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError(f"Could not load image: {file_path}")
        _, img_encoded = cv2.imencode(".jpg", frame)
        return img_encoded
    except Exception as e:
        logging.error(f"Error in image_searcher: {e}")
        return None


def image_crop(filepath):
    """Crop face from image with padding"""
    try:
        face_handler = FaceAnalysis(
            'antelopev2', 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
            root='.'
        )
        face_handler.prepare(ctx_id=0)
        
        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not load image: {filepath}")
            
        faces = face_handler.get(frame)
        if not faces:
            raise ValueError("No faces detected in image")
            
        facebox = faces[0].bbox
        x1, y1, x2, y2 = map(int, facebox)
        
        height_f, width_f = frame.shape[:2]
        padding = 40
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, width_f)
        y2 = min(y2 + padding, height_f)
        
        cropped_frame = frame[y1:y2, x1:x2]
        _, img_encoded = cv2.imencode(".jpg", cropped_frame)
        return img_encoded
        
    except Exception as e:
        logging.error(f"Error in image_crop: {e}")
        return None


if __name__ == "__main__":
    result = image_crop(r'dbimage\aref\image.png')
    if result is not None:
        print("Image cropped successfully")
    else:
        print("Failed to crop image")