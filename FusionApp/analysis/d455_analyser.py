import multiprocessing
import numpy as np
from queue import Empty
from camera.d455 import D455Frame
from typing import List, Optional
import time
import logging
from ultralytics import YOLO

from engine.interfaces import CameraAnalyser
from config_params import CFGS
from utils import setup_logger


class Rectangle:
    def __init__(self, x: float, y: float, width: float, height: float, class_id: int = 0, confidence: float = 1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.confidence = confidence
        
        # Map YOLO class IDs to names
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle'
        }
        
        self.object_type = self.class_names.get(class_id, 'unknown')


D455Objects = List[Rectangle]


class D455Results:
    def __init__(self, d455_frame: D455Frame, objects: D455Objects):
        self.objects = objects
        self.frame = d455_frame


class D455Analyser(CameraAnalyser):
    def __init__(self):
        # Initialize these in run() method
        self._yolo_model: Optional[YOLO] = None
        self.logger: Optional[logging.Logger] = None

    def _detect_objects(self, rgb_image: np.ndarray):
        """Detect persons, cars, bicycles, and motorcycles in the image"""
        objects = []
        try:
            if self._yolo_model is None:
                self._yolo_model = YOLO('yolov8n.pt')
            start_time = time.perf_counter()
            results = self._yolo_model(rgb_image, verbose=False)
            height, width, _ = rgb_image.shape
            
            # Target classes: person, bicycle, car, motorcycle
            target_classes = {0, 1, 2, 3}
            
            for r in results:
                for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    # Filter by class and confidence
                    if class_id in target_classes and confidence > 0.5:
                        x1, y1, x2, y2 = box
                        x = int(x1)
                        y = int(y1)
                        box_width = int(x2 - x1)
                        box_height = int(y2 - y1)
                        objects.append(Rectangle(x, y, box_width, box_height, class_id, confidence))
                        
            end_time = time.perf_counter()
            detection_time = end_time - start_time
            # self.logger.debug(f"YOLOv8 detection: {detection_time:.4f} seconds ({1/detection_time:.2f} FPS)")
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"YOLOv8 error: {e}")
        return objects

    def _analyse_frame(self, video_frame: D455Frame):
        rgb_image = video_frame.rgb_image
        objects = self._detect_objects(rgb_image)
        return objects

    def run(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        stop_event,
    ):
        # Set up logger in the child process
        self.logger = setup_logger("D455Analyser")
        self.logger.info("Starting D455Analyser...")
        # Load YOLOv8 model (uses GPU if available)
        self._yolo_model = YOLO('yolov8n.pt')
        
        # Performance tracking
        frame_count = 0
        total_analysis_time = 0
        total_processing_time = 0

        while not stop_event.is_set():
            try:
                total_start_time = time.perf_counter()
                
                # Add debug logging to see if we're receiving frames
                self.logger.debug("D455Analyser waiting for frame...")
                video_frame = input_queue.get(timeout=1)
                self.logger.debug(f"D455Analyser received frame with timestamp: {video_frame.timestamp}")
                
                # Track frame analysis time separately
                analysis_start_time = time.perf_counter()
                objects = self._analyse_frame(video_frame)
                analysis_end_time = time.perf_counter()
                analysis_time = analysis_end_time - analysis_start_time
                
                output_queue.put(D455Results(video_frame, objects))
                self.logger.debug(f"Put detection result in output queue: frame timestamp={video_frame.timestamp}, detected_objects={len(objects)}")
                total_end_time = time.perf_counter()
                total_processing_time_frame = total_end_time - total_start_time
                
                # Update statistics
                frame_count += 1
                total_analysis_time += analysis_time
                total_processing_time += total_processing_time_frame
                
                # Print average statistics every 30 frames at INFO level
                if frame_count % 30 == 0:
                    avg_analysis_time = total_analysis_time / frame_count
                    avg_total_time = total_processing_time / frame_count
                    self.logger.info(f"[CAMERA_PROFILE] Average over {frame_count} frames: analysis={avg_analysis_time:.4f}s ({1/avg_analysis_time:.2f} FPS), total={avg_total_time:.4f}s ({1/avg_total_time:.2f} FPS)")
                
                # Also log individual frame processing
                self.logger.debug(f"[CAMERA_PROFILE] Frame {frame_count}: analysis={analysis_time:.4f}s, total={total_processing_time_frame:.4f}s, detected {len(objects)} objects")
                    
            except Empty:
                self.logger.debug("D455Analyser: No frames available, continuing...")
                continue
            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received, stopping...")
                stop_event.set()
            except Exception as e:
                self.logger.error(f"Error processing frame {frame_count}: {e}")
                continue

        # Final summary
        if frame_count > 0:
            avg_analysis_time = total_analysis_time / frame_count
            avg_total_time = total_processing_time / frame_count
            self.logger.info(f"[CAMERA_PROFILE] Final summary: processed {frame_count} frames, avg analysis={avg_analysis_time:.4f}s ({1/avg_analysis_time:.2f} FPS), avg total={avg_total_time:.4f}s ({1/avg_total_time:.2f} FPS)")
        
        self.logger.info("D455Analyser stopped.")
