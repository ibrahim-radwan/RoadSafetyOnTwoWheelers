import multiprocessing
import os
import numpy as np
from queue import Empty, Full
from camera.d455 import D455Frame
from typing import List, Optional
import time
import logging
from ultralytics import YOLO
import torch

from engine.interfaces import CameraAnalyser
from config_params import CFGS
from utils import setup_logger


class Rectangle:
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        class_id: int = 0,
        confidence: float = 1.0,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.confidence = confidence

        # Map YOLO class IDs to names
        self.class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}

        self.object_type = self.class_names.get(class_id, "unknown")


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
        self._device: str = "cpu"
        self._half: bool = False
        self._imgsz: int = 480
        self._device_arg = "cpu"
        self._analysis_enabled: bool = True

        # Optimization status flags (for logging)
        self._opt_cuda_available: bool = False
        self._opt_cudnn_benchmark: bool = False
        self._opt_tf32_allowed: bool = False
        self._opt_matmul_precision_set: bool = False
        self._opt_device_move_ok: bool = False
        self._opt_channels_last_ok: bool = False
        self._opt_model_fused: bool = False
        self._opt_model_half: bool = False
        self._opt_predict_kwargs_supported: Optional[bool] = None
        self._opt_predict_kwargs_status_logged: bool = False

    def _detect_objects(self, rgb_image: np.ndarray):
        """Detect persons, cars, bicycles, and motorcycles in the image"""
        objects = []
        try:
            # Deactivate detection if hardware acceleration is not available
            if not self._analysis_enabled:
                return objects
            if self._yolo_model is None:
                self._yolo_model = YOLO("yolov8n.pt")
            start_time = time.perf_counter()
            # Simpler call path observed to be faster in your setup
            results = self._yolo_model(rgb_image, verbose=False)
            height, width, _ = rgb_image.shape

            # Target classes: person, bicycle, car, motorcycle
            target_classes = {0, 1, 2, 3}

            for r in results:
                for box, cls, conf in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    class_id = int(cls)
                    confidence = float(conf)

                    # Filter by class and confidence
                    if class_id in target_classes and confidence > 0.5:
                        x1, y1, x2, y2 = box
                        x = int(x1)
                        y = int(y1)
                        box_width = int(x2 - x1)
                        box_height = int(y2 - y1)
                        objects.append(
                            Rectangle(x, y, box_width, box_height, class_id, confidence)
                        )

            end_time = time.perf_counter()
            detection_time = end_time - start_time
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

        # Configure PyTorch/YOLO runtime for Jetson and check acceleration
        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._opt_cuda_available = self._device == "cuda"
            try:
                torch.backends.cudnn.benchmark = True
                self._opt_cudnn_benchmark = (
                    True if torch.backends.cudnn.benchmark else False
                )
                # Prefer TF32 on Ampere+ (e.g., Orin) when available
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    self._opt_tf32_allowed = True
                except Exception:
                    self._opt_tf32_allowed = False
                try:
                    torch.set_float32_matmul_precision("high")
                    self._opt_matmul_precision_set = True
                except Exception:
                    self._opt_matmul_precision_set = False
            except Exception:
                pass
            self._half = self._device == "cuda"
            self._device_arg = 0 if self._device == "cuda" else "cpu"
        except Exception:
            self._device = "cpu"
            self._half = False
            self._device_arg = "cpu"

        # Deactivate analysis if no CUDA and log an error once
        if self._device != "cuda":
            self._analysis_enabled = False
            self.logger.error(
                "GPU/HW acceleration is not available. Object detection is deactivated."
            )
            self.logger.info(
                f"[YOLO_OPT] cuda_available={self._opt_cuda_available}, cudnn_benchmark={self._opt_cudnn_benchmark}, tf32_enabled={self._opt_tf32_allowed}, matmul_precision_high={self._opt_matmul_precision_set}, imgsz={self._imgsz}"
            )
        else:
            # Load YOLOv8 model (GPU path only); rely on Ultralytics internals for placement/precision
            self._yolo_model = YOLO("yolov8n.pt")
            # Log device info
            try:
                name = torch.cuda.get_device_name(0)
                self.logger.info(
                    f"Using CUDA device: {name}, imgsz={self._imgsz}, half={self._half}"
                )
            except Exception:
                pass
            self.logger.info(
                f"[YOLO_OPT] cuda_available={self._opt_cuda_available}, cudnn_benchmark={self._opt_cudnn_benchmark}, tf32_enabled={self._opt_tf32_allowed}, matmul_precision_high={self._opt_matmul_precision_set}, optimizations='default-call'"
            )

            # Warm up kernels to reduce first-frame latency
            try:
                dummy = np.zeros((self._imgsz, 640, 3), dtype=np.uint8)
                for _ in range(3):
                    _ = self._yolo_model(dummy, verbose=False)
            except Exception:
                pass

        # Performance tracking
        frame_count = 0
        total_analysis_time = 0
        total_processing_time = 0
        total_queue_wait_time = 0
        total_output_put_time = 0

        while not stop_event.is_set():
            try:
                # Measure queue wait separately (not part of total processing time)
                queue_wait_start = time.perf_counter()
                video_frame = input_queue.get(timeout=1)
                queue_wait_end = time.perf_counter()
                queue_wait_time = queue_wait_end - queue_wait_start
                # Allow immediate shutdown via sentinel
                if isinstance(video_frame, dict) and video_frame.get("STOP"):
                    break
                total_start_time = time.perf_counter()

                # Track frame analysis time separately
                analysis_start_time = time.perf_counter()
                objects = self._analyse_frame(video_frame)
                analysis_end_time = time.perf_counter()
                analysis_time = analysis_end_time - analysis_start_time

                # Avoid blocking if consumer is stopping
                try:
                    output_put_start = time.perf_counter()
                    output_queue.put_nowait(D455Results(video_frame, objects))
                    output_put_end = time.perf_counter()
                    output_put_time = output_put_end - output_put_start
                except Full:
                    self.logger.warning(
                        "Camera results queue full, dropping result to exit cleanly"
                    )
                    output_put_time = 0.0
                    # Continue without blocking
                total_end_time = time.perf_counter()
                total_processing_time_frame = total_end_time - total_start_time

                # Update statistics
                frame_count += 1
                total_analysis_time += analysis_time
                total_processing_time += total_processing_time_frame
                total_queue_wait_time += queue_wait_time
                total_output_put_time += output_put_time

                # Print average statistics every 30 frames at INFO level
                if frame_count % 30 == 0:
                    avg_analysis_time = total_analysis_time / frame_count
                    avg_total_time = total_processing_time / frame_count
                    self.logger.info(
                        f"[CAMERA_PROFILE] Average over {frame_count} frames: analysis={avg_analysis_time:.4f}s ({1/avg_analysis_time:.2f} FPS), total={avg_total_time:.4f}s ({1/avg_total_time:.2f} FPS)"
                    )
                    # Provide additional breakdown at DEBUG level
                    avg_queue_wait = total_queue_wait_time / frame_count
                    avg_output_put = total_output_put_time / frame_count
                    self.logger.info(
                        f"[CAMERA_PROFILE_BREAKDOWN] queue_wait={avg_queue_wait:.4f}s, output_put={avg_output_put:.4f}s"
                    )

                # Also log individual frame processing
                self.logger.debug(
                    f"[CAMERA_PROFILE] Frame {frame_count}: analysis={analysis_time:.4f}s, total={total_processing_time_frame:.4f}s, queue_wait={queue_wait_time:.4f}s, output_put={output_put_time:.4f}s, detected {len(objects)} objects"
                )

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
            self.logger.info(
                f"[CAMERA_PROFILE] Final summary: processed {frame_count} frames, avg analysis={avg_analysis_time:.4f}s ({1/avg_analysis_time:.2f} FPS), avg total={avg_total_time:.4f}s ({1/avg_total_time:.2f} FPS)"
            )

        self.logger.info("D455Analyser stopped.")

        return
