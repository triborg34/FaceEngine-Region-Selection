import asyncio
from asyncio import Queue
import gc
import logging
import multiprocessing
import os
import platform
import queue
import subprocess
import sys
import time
import threading
import requests
from torchvision.models import resnet50
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
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
import json

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
        self.start()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frps = 5 if self.device == 'cuda' else 25
        self.MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
        self.TARGET_FPS = 30
        self.FRAME_DELAY = 1.0 / self.TARGET_FPS
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        # self.start()
        self.score,self.padding,self.quality=self.loadConfig()[0:3]

        # Initialize models
        self.model = None
        self.face_handler = None
        self._load_models()
        self.known_names = self.load_db()
        # Load database
        


        # Threading and process management
        self.process = None
        self.lock = threading.Lock()
        self.recognition_queue = queue.Queue()
        self.face_info = {}
        self.face_info_lock = threading.Lock()
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()

        # Image Searcher
        self.FOLDER_PATH = "outputs/humancrop"             # folder containing all images
        self.EMBEDDING_FILE = "embeddings.npy"  # file to save/load embeddings
        self.FILENAMES_FILE = "filenames.txt"  # file to save/load filenames
        self.LOCAL_WEIGHTS = "models/resnet50-0676ba61.pth"
        self.IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # regions
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.k = []

    def load_regions(self, soruce, file_path='regions.json',):
        url = urlparse(soruce).hostname
        """Load regions from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)


                if url == data['ip']:
                    return data.get('regions', {})
                else:
                    pass

        except Exception as e:
            print(f"Error loading regions: {e}")
            return {}

    def draw_regions_on_frame(self, frame,regions):
        """Draw region boundaries on frame"""
        overlay = frame.copy()

        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            color_name = region_data.get('color', 'red')
            shape_type = region_data.get('shape_type', 'polygon')

            # Convert color name to BGR
            color_map = {
                'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
                'yellow': (0, 255, 255), 'purple': (128, 0, 128),
                'orange': (0, 165, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255)
            }
            color = color_map.get(color_name, (0, 0, 255))

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 2)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

            # Add region label
            if points:
                center_x = int(sum(p[0] for p in points) / len(points))
                center_y = int(sum(p[1] for p in points) / len(points))

                # Add background for text
                text = f"{region_name} (ID: {region_data.get('id', 'N/A')})"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                # cv2.rectangle(overlay, (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                #               (center_x + text_size[0]//2 + 5, center_y + 5), (0, 0, 0), -1)
                # cv2.putText(overlay, text, (center_x - text_size[0]//2, center_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return overlay

    def get_detection_region(self, detection_box,region_masks):

        cx = int((detection_box[0] + detection_box[2]) / 2)
        cy = int((detection_box[1] + detection_box[3]) / 2)
        for region_name, mask in region_masks.items():

            if cy < mask.shape[0] and cx < mask.shape[1] and mask[cy, cx] > 0:
                return region_name  # First match wins
        return None

    def generate_region_masks(self, frame_shape,regions):
        """Create binary masks for each region (once)"""
        h, w, _ = frame_shape
        masks = {}
        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            shape_type = region_data.get('shape_type', 'polygon')

            mask = np.zeros((h, w), dtype=np.uint8)

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)  # use thickness

            masks[region_name] = mask
        return masks

    def onDisplay(self, region, frame):
        """Display region names on frame"""
        if not region:  # More pythonic than len(region) == 0
            return

        # Display up to the first few regions with proper spacing
        y_offset = 30  # Starting Y position
        line_height = 50  # Space between lines

        # Limit to 5 regions to avoid overcrowding
        for i, reg in enumerate(region[:5]):
            if 'name' in reg:
                y_pos = y_offset + (i * line_height)
                cv2.putText(frame, reg['name'], (10, y_pos),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

        # print(f"Active regions: {[r.get('name', 'Unknown') for r in region]}")
        # Handle relay operations in background
        if region:  # Only trigger if there are active regions
            threading.Thread(
                target=self.handle_relay_operations,
                args=[region.copy()],  # Pass a copy to avoid race conditions
                daemon=True
            ).start()

    def handle_relay_operations(self, regions):
        """Handle IP relay operations in background thread"""
        try:
            # Your IP relay operations here
            for region in regions:
                region_name = region.get('name', 'Unknown')
                # print(f"Background: Processing relay for {region_name}")

                # Example relay operations:
                # self.open_ip_relay(region_name)
                # time.sleep(0.1)  # Small delay between operations

                # Add your actual relay control code here
                # Example:
                # if region_name == 'R1':
                #     self.control_relay('192.168.1.100', 'open')
                # elif region_name == 'R2':
                #     self.control_relay('192.168.1.101', 'open')

        except Exception as e:
            print(f"Error in relay operations: {e}")
# Additional helper method for actual relay control

    def control_relay(self, ip_address, action, port=80, timeout=5):
        """Control IP relay - implement your specific relay protocol"""
        try:
            import requests  # or whatever library you use for relay control

            # Example for HTTP-based relay control
            url = f"http://{ip_address}/relay/{action}"
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                print(f"Successfully {action} relay at {ip_address}")
                return True
            else:
                print(
                    f"Failed to {action} relay at {ip_address}: {response.status_code}")
                return False

        except Exception as e:
            print(f"Error controlling relay {ip_address}: {e}")
            return False
# Example usage with specific relay mapping

    def setup_relay_mapping(self):
        """Setup mapping between regions and relay IPs"""
        self.relay_mapping = {
            'R1': {'ip': '192.168.1.100', 'port': 80},
            'R2': {'ip': '192.168.1.101', 'port': 80},
            'R3': {'ip': '192.168.1.102', 'port': 80},
            # Add more as needed
        }

    def loadConfig(self):
        uri='http://127.0.0.1:8091/api/collections/setting/records'
        response=requests.get(uri)
        data=response.json().get('items')[0]
        
        return float(data['score']),data['padding'],int(data['quality']),data['port']
    
    def load_image_searcher_model(self):
        model = resnet50(weights=None)  # don't load default
        # load weights from file
        state_dict = torch.load(self.LOCAL_WEIGHTS, map_location=self.device)
        model.load_state_dict(state_dict)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        return model

    def get_embedding(self, img_path):
        model = self.load_image_searcher_model()
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = model(img_t)
        features = features.view(features.size(0), -1).cpu().numpy().flatten()
        return features / np.linalg.norm(features)
  
    def precompute_embeddings(self, model, folder_path):
        logging.info("Precomputing embeddings for all images in folder...")
        embeddings = []
        filenames = []
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(self.IMG_EXTENSIONS):
                continue
            fpath = os.path.join(folder_path, fname)
            emb = self.get_embedding(fpath)
            embeddings.append(emb)
            filenames.append(fname)
            logging.info(f"Processed {fname}")
        embeddings = np.array(embeddings)
        np.save(self.EMBEDDING_FILE, embeddings)
        with open(self.FILENAMES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(filenames))
        logging.info(
            f"Saved embeddings to {self.EMBEDDING_FILE} and filenames to {self.FILENAMES_FILE}")
        return embeddings, filenames  

    def load_embeddings(self):
        embeddings = np.load(self.EMBEDDING_FILE)
        with open(self.FILENAMES_FILE, "r", encoding='utf-8') as f:
            filenames = f.read().splitlines()
        logging.info(f"Loaded {len(filenames)} embeddings from disk")
        return embeddings, filenames
 
    def find_similar_images(self, query_embedding, embeddings, filenames, top_k=10):
        sims = cosine_similarity([query_embedding], embeddings)[0]
        sorted_indices = np.argsort(sims)[::-1]
        results = [(filenames[i], sims[i]) for i in sorted_indices[:top_k]]
        return results
 
 
 
    def _load_models(self):
        """Load YOLO and face recognition models"""
        try:
            logging.info("Loading models...")

            # Load face handler
            self.face_handler = FaceAnalysis(#TODO:Change This
                'buffalo_l',
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

    def start(self):
        process = subprocess.Popen(
                        ["pocketbase", "serve", "--http=0.0.0.0:8091"], creationflags=subprocess.CREATE_NO_WINDOW)
        logging.info(f"PocketBase stater {process.pid}")

    def load_db(self):
        """Load known faces from database"""
        try:
            known_names = load_embeddings_from_db()
            logging.info(
                f"Loaded {len(known_names)} known faces from database")
            return known_names
        except Exception as e:
            logging.error(f"Failed to load database: {e}")
            return {}
   
    async def release_resources(self, fresh: FreshestFrame, cap: cv2.VideoCapture, role: bool):
        """Properly release camera resources"""
        try:
            if fresh:
                fresh.release()
            if cap:
                cap.release()
            # Signal recognition worker to stop
            if not role:
                self.recognition_queue.put(None)
                await self.stop_background_processing()
        except Exception as e:
            logging.error(f"Error releasing camera resources: {e}")

    async def graceful_shutdown(self):
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

                frame,path,track_id, face_img = item
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
                    det_score = float(face.det_score)
                    name, sim, gender, age, role = self.recognize_face(
                        face.embedding, gender, age
                    )
                    x1, y1, x2, y2 = map(int, face.bbox)

                    self.update_face_info(
                        track_id, name, sim, gender, age, role, (
                            x1, y1, x2, y2)
                    )
                    self.embedding_cache[track_id] = face.embedding
                else:
                    self.update_face_info(
                        track_id, "Unknown", 0.0, 'None', 'None', '', None
                    )

                if det_score>self.score:
                    height_f, width_f = face_img.shape[:2]
                    padding = self.padding
                    fx1_padded = max(x1 - padding, 0)
                    fy1_padded = max(y1 - padding, 0)
                    fx2_padded = min(x2 + padding, width_f)
                    fy2_padded = min(y2 + padding, height_f)

                    cropped_face = face_img[fy1_padded:fy2_padded,
                                                    fx1_padded:fx2_padded]
                    
                    try:
                        insertToDb(name,frame.copy(),cropped_face.copy(),face_img.copy(),det_score,track_id,gender,age,role,path,self.quality) #TODO
                    except Exception as e:
                                logging.error(f"Error inserting to DB: {e}")
                
                    

            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                logging.error(f"Error in recognition worker: {e}")

        logging.info("Recognition worker stopped.")

    async def process_frame(self, frame, path, counter,regions):
        """Process a single frame for object detection and face recognition"""
        try:
            if frame.size == 0:
                return frame

            start_time = time.time()

            # Resize frame for processing
            processed_frame = cv2.resize(frame, (1000, 1000))
            region_masks=self.generate_region_masks(processed_frame.shape,regions)
            combined_mask = np.zeros(processed_frame.shape[:2], dtype=np.uint8)
            for mask in region_masks.values():
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            masked_frame = cv2.bitwise_and(
                processed_frame, processed_frame, mask=combined_mask)
            self.k.clear()
            current_regions = []

            # Run YOLO detection
            results = self.model.track(
                masked_frame,
                classes=[0],  # Person class
                tracker="bytetrack.yaml",
                persist=True,
                device=self.device,
                conf=0.7 #TODO:GET CONF IN SETTING
            )

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4].cpu().tolist())
                    region_name = self.get_detection_region((x1, y1, x2, y2),region_masks)
                    if region_name and region_name in regions:
                        region_data = regions[region_name]
                        if region_data not in current_regions:
                            current_regions.append(region_data)

                    # Get tracking ID
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].cpu().item())

                    # Crop human region
                    human_crop = masked_frame[y1:y2, x1:x2]
                    if human_crop.size == 0:
                        continue

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    # Queue for recognition every frps frames
                    # if counter % self.frps == 0:
                    self.recognition_queue.put((processed_frame.copy(),path,track_id, human_crop))

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

                    # try:
                    #     score = int(info['score'] *
                    #                 100) if info['score'] else 0
                    # except (TypeError, ValueError):
                    #     score = 0

                    # name = info['name']
                    # gender = info['gender']
                    # age = info['age']
                    # role = info['role']

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
                        # height_f, width_f = human_crop.shape[:2]
                        # padding = 40
                        # fx1_padded = max(fx1 - padding, 0)
                        # fy1_padded = max(fy1 - padding, 0)
                        # fx2_padded = min(fx2 + padding, width_f)
                        # fy2_padded = min(fy2 + padding, height_f)

                        # cropped_face = human_crop[fy1_padded:fy2_padded,
                        #                           fx1_padded:fx2_padded]

                        # Insert to database
                        # try:
                        #     await insertToDb(
                        #         name, processed_frame, cropped_face, human_crop,
                        #         score, track_id, gender, age, role, path
                        #     )
                        # except Exception as e:
                        #     logging.error(f"Error inserting to DB: {e}")

                    else:
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

            # Calculate and display FPS
            self.k = current_regions
            self.onDisplay(self.k, processed_frame)
            display_frame = self.draw_regions_on_frame(processed_frame,regions)
            try:
                fps = 1.0 / (time.time() - start_time)
            except ZeroDivisionError:
                fps = 30
            cv2.putText(
                display_frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

            return display_frame

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

    async def generate_frames(self, camera_idx, source, request: Request, role: bool):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        check_interval = 60  # seconds
        last_check = 0
        counter = 0
        regions=self.load_regions(soruce=source)
        if regions ==None:
            regions={ "r2": {
      "id": "1345",
      "name": "r1",
      "description": "",
      "points": [
        [0.0, 0.0],          # top-left
  [999.0, 0.0],        #top-right
  [639.0, 999.0],      #bottom-right
  [0.0, 999.0],       # bottom-left
  [0.0, 0.0] 
      ],
      "shape_type": "polygon",
      "color": "red",
      "created": "2025-08-05T11:46:12.379819",
      
      "ip":urlparse(source).hostname
    },}
        # logging.info("Regions loaded:", list(regions.keys()))
 

        if not hasattr(self, 'k'):
            self.k = []

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
                        logging.warning(
                            f"[Camera {camera_idx}] Connection lost")
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

                    if role == True:
                        frame = frame
                    # Process frame
                    else:
                        frame = await self.process_frame(frame, f'/rt{camera_idx}', counter,regions)

                    # Resize back to original dimensions
                    # frame = cv2.resize(
                    #     frame, (1920, 1080))


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


def image_crop(filepath, isSearch):
    """Crop face from image with padding"""
    if isSearch:
        frame = cv2.imread(filepath)
        _, img_encoded = cv2.imencode(".jpg", frame)
        return img_encoded
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
