"""Vision engine — webcam capture, YOLO detection, Moondream, pose, gaze, and identify."""

from __future__ import annotations

import base64
from collections import deque
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
from urllib.request import urlopen
import os

import cv2
import numpy as np

from . import config


class VisionEngine:
    """Background webcam capture with YOLO, gesture, posture, gaze, and identify."""

    def __init__(self) -> None:
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: List[str] = []
        # Store (label, x1, y1, x2, y2) for identify cropping
        self._latest_bboxes: List[Tuple[str, int, int, int, int]] = []
        self._annotated_frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._camera: Optional[cv2.VideoCapture] = None
        self._yolo_model = None
        self._detection_interval = config.YOLO_DETECTION_INTERVAL
        self._last_detection_time = 0.0
        self._label_hits: dict[str, int] = {}
        self._airdraw_enabled = config.AIRDRAW_DEFAULT_ENABLED
        self._airdraw_canvas: Optional[np.ndarray] = None
        self._airdraw_last_point: Optional[Tuple[int, int]] = None
        self._last_finger_count: Optional[int] = None
        self._finger_history: deque[int] = deque(maxlen=max(1, config.HAND_SMOOTHING_WINDOW))
        self._grab_fail_count = 0

        # ── Posture tracking ────────────────────────────────────────
        self._mp_pose = None
        self._pose_frame_counter = 0
        self._current_posture = "unknown"        # good | hunched | tilted
        self._posture_history: deque[str] = deque(maxlen=6)
        self._on_posture_change: Optional[Callable[[str], None]] = None
        self._last_posture_reported = "unknown"

        # ── Gaze / face-mesh tracking ───────────────────────────────
        self._mp_face_mesh = None
        self._gaze_frame_counter = 0
        self._current_gaze_region = "center"    # e.g. "bottom_right"

        # ── Identify gesture ────────────────────────────────────────
        self._identify_gesture_start: Optional[float] = None
        self._last_identify_time = 0.0
        self._on_identify: Optional[Callable[[str], None]] = None

        # ── Hand pipeline (unchanged) ────────────────────────────────
        self._hand_pipeline = "contour"
        self._mp_hands = None
        self._mp_draw = None
        self._mp_task_landmarker = None
        self._mp_image = None
        self._mp_image_format_srgb = None
        self._mp_running_mode = None
        self._mp_timestamp_ms = 0
        self._init_hand_pipeline()

        # ── Optional: pose + face mesh ──────────────────────────────
        if config.POSTURE_ENABLED:
            self._init_pose_pipeline()
        if config.GAZE_ENABLED:
            self._init_gaze_pipeline()

        self._init_camera()

    # ── Pose pipeline (posture correction) ───────────────────────────

    def _init_pose_pipeline(self) -> None:
        """Initialise pose landmarker — Tasks API first, solutions fallback."""
        # ── Tasks API (Python 3.13 / mediapipe >= 0.10) ──────────────
        try:
            from mediapipe.tasks import python as _mp_tasks  # type: ignore
            from mediapipe.tasks.python import vision as _mp_vis  # type: ignore

            model_path = self._ensure_pose_task_model()
            options = _mp_vis.PoseLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=_mp_vis.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_pose = _mp_vis.PoseLandmarker.create_from_options(options)
            self._mp_pose_ts = 0
            self._mp_pose_api = "tasks"
            print("[Vision] Pose pipeline: mediapipe tasks pose (lite)")
            return
        except Exception as e:
            print(f"[Vision] Pose Tasks unavailable: {e}")

        # ── Legacy solutions API ──────────────────────────────────────
        try:
            import mediapipe as mp  # type: ignore
            if not hasattr(mp, "solutions"):
                raise RuntimeError("mediapipe.solutions unavailable")
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_pose_api = "solutions"
            print("[Vision] Pose pipeline: mediapipe solutions pose (lite)")
        except Exception as e:
            print(f"[Vision] Pose pipeline unavailable: {e}")
            self._mp_pose = None
            self._mp_pose_api = "none"

    def _ensure_pose_task_model(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, ".models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
        if os.path.exists(model_path):
            return model_path
        url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        )
        print("[Vision] Downloading pose landmark model …")
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as r:
            data = r.read()
        with open(model_path, "wb") as f:
            f.write(data)
        print(f"[Vision] Pose model ready: {model_path}")
        return model_path

    # ── Face-mesh / gaze pipeline ────────────────────────────────────

    def _init_gaze_pipeline(self) -> None:
        """Initialise face landmarker for gaze — Tasks API first, solutions fallback."""
        # ── Tasks API ────────────────────────────────────────────────
        try:
            from mediapipe.tasks import python as _mp_tasks  # type: ignore
            from mediapipe.tasks.python import vision as _mp_vis  # type: ignore

            model_path = self._ensure_face_task_model()
            options = _mp_vis.FaceLandmarkerOptions(
                base_options=_mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=_mp_vis.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_face_mesh = _mp_vis.FaceLandmarker.create_from_options(options)
            self._mp_face_ts = 0
            self._mp_face_api = "tasks"
            print("[Vision] Gaze pipeline: mediapipe tasks face landmarker")
            return
        except Exception as e:
            print(f"[Vision] Gaze Tasks unavailable: {e}")

        # ── Legacy solutions API ──────────────────────────────────────
        try:
            import mediapipe as mp  # type: ignore
            if not hasattr(mp, "solutions"):
                raise RuntimeError("mediapipe.solutions unavailable")
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_face_api = "solutions"
            print("[Vision] Gaze pipeline: mediapipe solutions face mesh (iris)")
        except Exception as e:
            print(f"[Vision] Gaze pipeline unavailable: {e}")
            self._mp_face_mesh = None
            self._mp_face_api = "none"

    def _ensure_face_task_model(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, ".models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        if os.path.exists(model_path):
            return model_path
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
        print("[Vision] Downloading face landmark model …")
        from urllib.request import urlopen
        with urlopen(url, timeout=60) as r:
            data = r.read()
        with open(model_path, "wb") as f:
            f.write(data)
        print(f"[Vision] Face model ready: {model_path}")
        return model_path

    # ── Posture & gaze helpers ───────────────────────────────────────

    def _estimate_posture(self, frame: np.ndarray) -> str:
        """Return 'good', 'hunched', or 'tilted' based on pose landmarks."""
        if self._mp_pose is None:
            return "unknown"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = None

        api = getattr(self, "_mp_pose_api", "solutions")
        if api == "tasks":
            try:
                import mediapipe as mp  # type: ignore
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._mp_pose_ts += 33
                result = self._mp_pose.detect_for_video(mp_img, self._mp_pose_ts)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]  # list of NormalizedLandmark
            except Exception:
                return "unknown"
        else:
            result = self._mp_pose.process(rgb)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

        if lm is None:
            return "unknown"

        nose = lm[0]
        ls = lm[11]
        rs = lm[12]
        shoulder_mid_y = (ls.y + rs.y) / 2.0
        shoulder_diff = abs(ls.y - rs.y)
        head_gap = shoulder_mid_y - nose.y

        if shoulder_diff > config.POSTURE_SHOULDER_TILT_THRESHOLD:
            return "tilted"
        if head_gap < config.POSTURE_HEAD_DROP_THRESHOLD:
            return "hunched"
        return "good"

    def _estimate_gaze(self, frame: np.ndarray) -> str:
        """Return screen region string based on face/iris position in frame.

        Screen is divided into a 3×3 grid:
            top_left  top_center  top_right
            left      center      right
            bot_left  bot_center  bot_right
        """
        if self._mp_face_mesh is None:
            return "center"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm = None

        api = getattr(self, "_mp_face_api", "solutions")
        if api == "tasks":
            try:
                import mediapipe as mp  # type: ignore
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self._mp_face_ts += 33
                result = self._mp_face_mesh.detect_for_video(mp_img, self._mp_face_ts)
                if result.face_landmarks:
                    lm = result.face_landmarks[0]  # list of NormalizedLandmark
            except Exception:
                return "center"
        else:
            result = self._mp_face_mesh.process(rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark

        if lm is None:
            return "center"

        nose_x = lm[4].x
        nose_y = lm[4].y

        # Iris refinement if landmarks are available (solutions with refine=True,
        # or Tasks FaceLandmarker which always includes them)
        try:
            iris_x = (lm[468].x + lm[473].x) / 2.0
            iris_y = (lm[468].y + lm[473].y) / 2.0
            nose_x = 0.4 * nose_x + 0.6 * iris_x
            nose_y = 0.4 * nose_y + 0.6 * iris_y
        except (IndexError, AttributeError):
            pass

        col = "left" if nose_x < 0.38 else ("right" if nose_x > 0.62 else "center")
        row = "top" if nose_y < 0.35 else ("bot" if nose_y > 0.65 else "")

        if row and col != "center":
            return f"{row}_{col}"
        if row:
            return f"{row}_center"
        return col

    # ── Identify object via Moondream ────────────────────────────────

    def identify_object(self) -> str:
        """Crop the most-detected object (or frame centre) and ask Moondream."""
        frame = self.get_latest_frame()
        if frame is None:
            return "Camera is not available, sir."

        h, w = frame.shape[:2]
        with self._frame_lock:
            bboxes = list(self._latest_bboxes)

        if bboxes:
            # Pick the largest bounding box by area
            best = max(bboxes, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))
            _, x1, y1, x2, y2 = best
            # Add 20 % padding
            pad_x = int((x2 - x1) * 0.20)
            pad_y = int((y2 - y1) * 0.20)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
        else:
            # Use centre 50 % of frame
            x1, y1 = int(w * 0.25), int(h * 0.25)
            x2, y2 = int(w * 0.75), int(h * 0.75)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "Could not crop a valid region for identification, sir."

        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return "Failed to encode image for identification, sir."
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        payload = {
            "model": config.VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Identify this object precisely. "
                        "If it is electronic hardware (microcontroller, sensor, module, PCB), "
                        "state the exact model or chip name. "
                        "Keep your answer concise, under two sentences."
                    ),
                    "images": [image_b64],
                }
            ],
        }

        url = f"{config.OLLAMA_URL}/api/chat"
        import json as _json

        # ── Try httpx first, fall back to urllib ──────────────────
        try:
            import httpx
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                return content if content else "I could not identify the object clearly."
        except ImportError:
            pass  # httpx not installed — fall through to urllib
        except Exception as exc:
            err = str(exc)
            if "404" in err or "not found" in err.lower():
                return (
                    f"Vision model '{config.VISION_MODEL}' is not installed. "
                    f"Run: ollama pull {config.VISION_MODEL}"
                )
            if any(k in err.lower() for k in ("connect", "refused", "unreachable")):
                return "Cannot reach Ollama. Make sure it is running: ollama serve"
            print(f"[Vision] identify httpx error: {exc}")
            return f"Identification failed: {exc}"

        # ── urllib fallback ───────────────────────────────────────
        try:
            from urllib.request import Request, urlopen
            body = _json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=60) as r:
                data = _json.loads(r.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "").strip()
            return content if content else "I could not identify the object clearly."
        except Exception as exc:
            err = str(exc)
            if "404" in err or "not found" in err.lower():
                return f"Vision model '{config.VISION_MODEL}' is not installed. Run: ollama pull {config.VISION_MODEL}"
            if any(k in err.lower() for k in ("connect", "refused", "unreachable")):
                return "Cannot reach Ollama. Make sure it is running: ollama serve"
            return f"Identification failed: {exc}"

    def _init_camera(self) -> None:
        print("[Vision] Initialising camera …")
        # Prefer DirectShow on Windows for better stability than MSMF in long-running loops.
        self._camera = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self._camera.isOpened():
            self._camera = cv2.VideoCapture(config.CAMERA_INDEX)
        if not self._camera.isOpened():
            print(f"[Vision] WARNING: Cannot open camera {config.CAMERA_INDEX}")
            self._camera = None
        else:
            # Set reasonable resolution
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("[Vision] Camera ready")

    def _init_hand_pipeline(self) -> None:
        desired = (config.HAND_PIPELINE or "auto").lower()
        if desired not in {"auto", "mediapipe", "contour"}:
            desired = "auto"

        # Prefer MediaPipe Tasks first (works on Python 3.13 builds where solutions API is missing).
        if desired in {"auto", "mediapipe"}:
            try:
                import mediapipe as mp  # type: ignore
                from mediapipe.tasks import python as mp_tasks_python  # type: ignore
                from mediapipe.tasks.python import vision as mp_tasks_vision  # type: ignore

                model_path = self._ensure_hand_task_model()
                base_options = mp_tasks_python.BaseOptions(model_asset_path=model_path)
                options = mp_tasks_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_tasks_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONF,
                    min_hand_presence_confidence=config.HAND_MIN_TRACKING_CONF,
                    min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
                )
                self._mp_task_landmarker = mp_tasks_vision.HandLandmarker.create_from_options(options)
                self._mp_image = mp.Image
                self._mp_image_format_srgb = mp.ImageFormat.SRGB
                self._mp_running_mode = mp_tasks_vision.RunningMode
                self._hand_pipeline = "mediapipe-tasks"
                print("[Vision] Hand pipeline: mediapipe tasks landmarks")
                return
            except Exception as e:
                if desired == "mediapipe":
                    print(f"[Vision] MediaPipe Tasks requested but unavailable: {e}")
                else:
                    print(f"[Vision] MediaPipe Tasks not available, trying legacy solutions: {e}")

        # Legacy solutions backend (older environments).
        if desired in {"auto", "mediapipe"}:
            try:
                import mediapipe as mp  # type: ignore

                if not hasattr(mp, "solutions"):
                    raise RuntimeError("mediapipe.solutions is not available in this build")

                hands_module = mp.solutions.hands
                self._mp_hands = hands_module.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=config.HAND_MIN_DETECTION_CONF,
                    min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
                )
                self._mp_draw = mp.solutions.drawing_utils
                self._hand_pipeline = "mediapipe"
                print("[Vision] Hand pipeline: mediapipe landmarks")
                return
            except Exception as e:
                if desired == "mediapipe":
                    print(f"[Vision] MediaPipe requested but unavailable: {e}")
                else:
                    print(f"[Vision] MediaPipe not available, using contour fallback: {e}")

        self._hand_pipeline = "contour"
        print("[Vision] Hand pipeline: contour fallback")

    def _ensure_hand_task_model(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, ".models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "hand_landmarker.task")
        if os.path.exists(model_path):
            return model_path

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        )
        print("[Vision] Downloading hand landmark model asset …")
        with urlopen(model_url, timeout=30) as resp:
            data = resp.read()
        with open(model_path, "wb") as f:
            f.write(data)
        print(f"[Vision] Hand landmark model ready: {model_path}")
        return model_path

    def _ensure_yolo(self):
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(config.YOLO_MODEL)
                print(f"[Vision] YOLO loaded: {config.YOLO_MODEL}")
            except Exception as e:
                print(f"[Vision] YOLO load failed: {e}")
        return self._yolo_model

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running or self._camera is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[Vision] Capture loop started")

    def stop(self) -> None:
        """Stop the capture thread and release camera."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._camera is not None:
            self._camera.release()
        if self._mp_hands is not None:
            try:
                self._mp_hands.close()
            except Exception:
                pass
        if self._mp_task_landmarker is not None:
            try:
                self._mp_task_landmarker.close()
            except Exception:
                pass
        print("[Vision] Stopped")

    def _capture_loop(self) -> None:
        """Continuously read frames and periodically run detection."""
        frame_idx = 0

        while self._running and self._camera is not None:
            ok, frame = self._camera.read()
            if not ok:
                self._grab_fail_count += 1
                if self._grab_fail_count >= 20:
                    self._reopen_camera()
                    self._grab_fail_count = 0
                time.sleep(0.05)
                continue

            self._grab_fail_count = 0
            frame_idx += 1

            now = time.time()
            detections: List[str] = []
            new_bboxes: List[Tuple[str, int, int, int, int]] = []
            annotated = frame.copy()
            frame_h, frame_w = frame.shape[:2]

            if self._airdraw_canvas is None or self._airdraw_canvas.shape[:2] != (frame_h, frame_w):
                self._airdraw_canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            # ── Hand + gesture ───────────────────────────────────────
            finger_count = None
            fingertip = None
            if config.GESTURE_ENABLED:
                finger_count, fingertip = self._estimate_hand_state(frame, annotated)
                if finger_count is not None:
                    self._finger_history.append(finger_count)
                    self._last_finger_count = int(round(float(np.median(np.array(list(self._finger_history), dtype=np.float32)))))

                if self._airdraw_enabled:
                    if finger_count == 1 and fingertip is not None and self._airdraw_canvas is not None:
                        if self._airdraw_last_point is None:
                            self._airdraw_last_point = fingertip
                        cv2.line(
                            self._airdraw_canvas,
                            self._airdraw_last_point,
                            fingertip,
                            (80, 255, 255),
                            config.AIRDRAW_BRUSH_THICKNESS,
                        )
                        self._airdraw_last_point = fingertip
                    else:
                        self._airdraw_last_point = None

            # ── Identify gesture: index-only hold ───────────────────
            if config.IDENTIFY_GESTURE_ENABLED and self._on_identify is not None:
                if self._last_finger_count == 1:
                    if self._identify_gesture_start is None:
                        self._identify_gesture_start = now
                    elif (
                        now - self._identify_gesture_start >= config.IDENTIFY_HOLD_SECONDS
                        and now - self._last_identify_time >= config.IDENTIFY_COOLDOWN
                    ):
                        self._last_identify_time = now
                        self._identify_gesture_start = None
                        import threading as _t
                        _t.Thread(
                            target=lambda: self._on_identify(self.identify_object()),
                            daemon=True,
                        ).start()
                else:
                    self._identify_gesture_start = None

            if self._airdraw_canvas is not None:
                mask = self._airdraw_canvas.sum(axis=2) > 0
                annotated[mask] = cv2.addWeighted(annotated, 0.45, self._airdraw_canvas, 0.55, 0)[mask]

            if self._last_finger_count is not None:
                cv2.putText(
                    annotated,
                    f"Fingers: {self._last_finger_count}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (90, 220, 255),
                    2,
                )

            # ── Posture check (every N frames) ───────────────────────
            if config.POSTURE_ENABLED and self._mp_pose is not None:
                self._pose_frame_counter += 1
                if self._pose_frame_counter >= config.POSTURE_FRAME_INTERVAL:
                    self._pose_frame_counter = 0
                    try:
                        posture = self._estimate_posture(frame)
                        if posture != "unknown":
                            self._posture_history.append(posture)
                            counts: Dict[str, int] = {}
                            for p in self._posture_history:
                                counts[p] = counts.get(p, 0) + 1
                            dominant = max(counts, key=lambda k: counts[k])
                            with self._frame_lock:
                                self._current_posture = dominant
                            # Fire callback on change (engine will rate-limit alerts)
                            if dominant != self._last_posture_reported:
                                self._last_posture_reported = dominant
                                if self._on_posture_change is not None:
                                    self._on_posture_change(dominant)
                    except Exception:
                        pass

            # ── Gaze estimate (every N frames) ───────────────────────
            if config.GAZE_ENABLED and self._mp_face_mesh is not None:
                self._gaze_frame_counter += 1
                if self._gaze_frame_counter >= config.GAZE_FRAME_INTERVAL:
                    self._gaze_frame_counter = 0
                    try:
                        region = self._estimate_gaze(frame)
                        with self._frame_lock:
                            self._current_gaze_region = region
                    except Exception:
                        pass

            # ── YOLO periodically ────────────────────────────────────
            if now - self._last_detection_time > self._detection_interval:
                try:
                    model = self._ensure_yolo()
                    if model is not None:
                        results = model.predict(
                            source=frame,
                            conf=config.YOLO_CONFIDENCE,
                            verbose=False,
                        )
                        labels = set()
                        current_hits: set[str] = set()
                        for result in results:
                            names = result.names
                            for box in result.boxes:
                                cls_idx = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                if conf >= config.YOLO_CONFIDENCE:
                                    label = names.get(cls_idx, str(cls_idx))
                                    if label in config.YOLO_IGNORE_LABELS:
                                        continue

                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    bw = max(1, x2 - x1)
                                    bh = max(1, y2 - y1)
                                    area_ratio = (bw * bh) / float(frame_w * frame_h)
                                    cx = (x1 + x2) / 2.0
                                    cy = (y1 + y2) / 2.0
                                    centered = (
                                        abs(cx - frame_w / 2.0) < frame_w * 0.22
                                        and abs(cy - frame_h / 2.0) < frame_h * 0.28
                                    )
                                    if label in config.YOLO_SUSPECT_ANIMAL_LABELS and centered and area_ratio > 0.30:
                                        continue

                                    current_hits.add(label)
                                    labels.add(label)
                                    new_bboxes.append((label, int(x1), int(y1), int(x2), int(y2)))

                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 200), 2)
                                    cv2.putText(
                                        annotated,
                                        f"{label} {conf:.0%}",
                                        (x1, y1 - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 200),
                                        1,
                                    )
                        next_hits: dict[str, int] = {}
                        for label in current_hits:
                            next_hits[label] = self._label_hits.get(label, 0) + 1
                        self._label_hits = next_hits

                        stable = [
                            label
                            for label in labels
                            if self._label_hits.get(label, 0) >= config.YOLO_MIN_STABLE_FRAMES
                        ]
                        stable_bboxes = [b for b in new_bboxes if b[0] in set(stable)]
                        detections = sorted(stable)
                        self._last_detection_time = now

                        with self._frame_lock:
                            self._latest_bboxes = stable_bboxes
                except Exception:
                    pass  # Don't crash the capture loop

            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._annotated_frame = annotated
                if detections:
                    self._latest_detections = detections

            time.sleep(0.03)  # ~30 fps

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest raw camera frame."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame with YOLO bounding boxes drawn."""
        with self._frame_lock:
            return self._annotated_frame.copy() if self._annotated_frame is not None else None

    def get_detections(self) -> List[str]:
        """Get the latest YOLO detection labels."""
        with self._frame_lock:
            return list(self._latest_detections)

    def get_finger_count(self) -> Optional[int]:
        with self._frame_lock:
            return self._last_finger_count

    def enable_airdraw(self, enabled: bool) -> None:
        with self._frame_lock:
            self._airdraw_enabled = enabled
            if not enabled:
                self._airdraw_last_point = None

    def clear_airdraw(self) -> None:
        with self._frame_lock:
            if self._airdraw_canvas is not None:
                self._airdraw_canvas[:] = 0
            self._airdraw_last_point = None

    def get_airdraw_canvas(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._airdraw_canvas.copy() if self._airdraw_canvas is not None else None

    def summarize_airdraw(self) -> str:
        canvas = self.get_airdraw_canvas()
        if canvas is None:
            return "Air-draw canvas is not available yet."

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "I do not see a drawing yet."

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        vertices = len(approx)
        shape = "freeform sketch"
        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            shape = "quadrilateral"
        elif vertices >= 8:
            shape = "circle-like shape"

        x, y, w, h = cv2.boundingRect(cnt)
        return (
            f"I can see a {shape} drawn in air. "
            f"Approx bounds are {w} by {h} pixels, with drawn area around {int(area)}."
        )

    def get_detection_context(self) -> str:
        """Get interesting objects only — skip obvious stuff."""
        labels = self.get_detections()
        # Filter out anything boring
        interesting = [l for l in labels if l not in config.YOLO_IGNORE_LABELS]
        if not interesting:
            return ""  # Nothing worth mentioning
        return ", ".join(interesting)

    def frame_as_base64(self) -> str:
        """Get the latest frame as a base64-encoded JPEG string."""
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("No frame available")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Failed to encode frame")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def describe_scene(self) -> str:
        """Use Moondream via Ollama to describe the current scene."""
        try:
            image_b64 = self.frame_as_base64()
        except RuntimeError:
            return "Camera is not available, sir."

        payload = {
            "model": config.VISION_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Describe what you see in this room. Mention people, objects, and the general setting.",
                    "images": [image_b64],
                }
            ],
        }

        url = f"{config.OLLAMA_URL}/api/chat"
        import json as _json

        def _parse(data: dict) -> str:
            return (data.get("message", {}).get("content", "") or "").strip() \
                or "I can see the scene but couldn't describe it."

        def _friendly_err(exc: Exception) -> str:
            err = str(exc)
            if "404" in err or "not found" in err.lower():
                return f"Vision model '{config.VISION_MODEL}' not installed. Run: ollama pull {config.VISION_MODEL}"
            if any(k in err.lower() for k in ("connect", "refused", "unreachable")):
                return "Cannot reach Ollama. Make sure it is running: ollama serve"
            return f"Scene analysis failed: {exc}"

        try:
            import httpx
            with httpx.Client(timeout=90.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return _parse(resp.json())
        except ImportError:
            pass
        except Exception as exc:
            return _friendly_err(exc)

        try:
            from urllib.request import Request, urlopen
            body = _json.dumps(payload).encode("utf-8")
            req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=90) as r:
                return _parse(_json.loads(r.read().decode("utf-8")))
        except Exception as exc:
            return _friendly_err(exc)

    def get_posture(self) -> str:
        with self._frame_lock:
            return self._current_posture

    def get_gaze_region(self) -> str:
        with self._frame_lock:
            return self._current_gaze_region

    @property
    def is_available(self) -> bool:
        return self._camera is not None and self._running

    def _reopen_camera(self) -> None:
        try:
            if self._camera is not None:
                self._camera.release()
            cam = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cam.isOpened():
                cam = cv2.VideoCapture(config.CAMERA_INDEX)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._camera = cam
            else:
                self._camera = None
        except Exception:
            self._camera = None

    def _estimate_hand_state(self, frame: np.ndarray, annotated: Optional[np.ndarray] = None) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        if self._hand_pipeline == "mediapipe-tasks":
            mp_task_result = self._estimate_hand_state_mediapipe_tasks(frame, annotated)
            if mp_task_result[0] is not None:
                return mp_task_result
        if self._hand_pipeline == "mediapipe":
            mp_result = self._estimate_hand_state_mediapipe(frame, annotated)
            if mp_result[0] is not None:
                return mp_result
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < config.AIRDRAW_MIN_HAND_AREA:
            return None, None

        fingertip = None
        top_idx = np.argmin(cnt[:, :, 1])
        if top_idx is not None:
            p = cnt[top_idx][0]
            fingertip = (int(p[0]), int(p[1]))

        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 4:
            return 1, fingertip

        defects = cv2.convexityDefects(cnt, hull_indices)
        if defects is None:
            return 1, fingertip

        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = cnt[s][0]
            end = cnt[e][0]
            far = cnt[f][0]

            a = np.linalg.norm(end - start)
            b = np.linalg.norm(far - start)
            c = np.linalg.norm(end - far)
            if b == 0 or c == 0:
                continue
            angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))
            if angle <= 90 and d > 6000:
                finger_gaps += 1

        count = max(1, min(5, finger_gaps + 1))
        return count, fingertip

    def _estimate_hand_state_mediapipe(
        self,
        frame: np.ndarray,
        annotated: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        if self._mp_hands is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, None

        hand_landmarks = result.multi_hand_landmarks[0]
        h, w = frame.shape[:2]

        if annotated is not None and config.HAND_DRAW_LANDMARKS:
            try:
                import mediapipe as mp  # type: ignore
                if self._mp_draw is not None:
                    self._mp_draw.draw_landmarks(
                        annotated,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        self._mp_draw.DrawingSpec(color=(50, 160, 255), thickness=1, circle_radius=2),
                        self._mp_draw.DrawingSpec(color=(120, 255, 200), thickness=1),
                    )
            except Exception:
                pass

        lm = hand_landmarks.landmark
        tip_ids = [4, 8, 12, 16, 20]

        # MediaPipe solutions already flips handedness labels to match user perspective
        handedness_label = "Right"
        if result.multi_handedness and len(result.multi_handedness) > 0:
            try:
                handedness_label = result.multi_handedness[0].classification[0].label
            except Exception:
                handedness_label = "Right"

        fingers_up = 0

        # Thumb
        thumb_tip = lm[tip_ids[0]]
        thumb_ip = lm[3]
        if handedness_label.lower() == "right":
            if thumb_tip.x < thumb_ip.x:
                fingers_up += 1
        else:
            if thumb_tip.x > thumb_ip.x:
                fingers_up += 1

        # Other four fingers: tip y strictly less than PIP y
        for tip_id in tip_ids[1:]:
            tip = lm[tip_id]
            pip = lm[tip_id - 2]
            if tip.y < pip.y - 0.01:  # small hysteresis prevents jitter
                fingers_up += 1

        index_tip = lm[8]
        fingertip = (int(index_tip.x * w), int(index_tip.y * h))
        fingers_up = max(0, min(5, fingers_up))
        return fingers_up, fingertip

    def _estimate_hand_state_mediapipe_tasks(
        self,
        frame: np.ndarray,
        annotated: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        if self._mp_task_landmarker is None or self._mp_image is None or self._mp_image_format_srgb is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image(image_format=self._mp_image_format_srgb, data=rgb)
        self._mp_timestamp_ms += 33
        result = self._mp_task_landmarker.detect_for_video(mp_image, self._mp_timestamp_ms)
        if not result.hand_landmarks:
            return None, None

        lms = result.hand_landmarks[0]
        h, w = frame.shape[:2]
        points = [(int(p.x * w), int(p.y * h)) for p in lms]

        if annotated is not None and config.HAND_DRAW_LANDMARKS:
            # Draw connection lines first
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17),
            ]
            for a, b in connections:
                cv2.line(annotated, points[a], points[b], (50, 160, 255), 1)
            # Draw landmark dots
            for idx, p in enumerate(points):
                r = 4 if idx in (4, 8, 12, 16, 20) else 2
                color = (120, 255, 200) if idx in (4, 8, 12, 16, 20) else (120, 200, 255)
                cv2.circle(annotated, p, r, color, -1)

        # ── Handedness (use model result if available, else heuristic) ──
        handedness_label = "Right"
        try:
            if result.handedness and result.handedness[0]:
                handedness_label = result.handedness[0][0].category_name  # "Left" or "Right"
        except Exception:
            # Fallback heuristic: in an UNMIRRORED camera view,
            # the right hand's wrist x > thumb_mcp x on average
            wrist = lms[0]
            mcp_middle = lms[9]
            handedness_label = "Right" if wrist.x > mcp_middle.x else "Left"

        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]

        fingers_up = 0

        # Thumb: compare tip x vs IP joint x, accounting for which hand
        thumb_tip = lms[4]
        thumb_ip = lms[3]
        if handedness_label.lower() == "right":
            if thumb_tip.x < thumb_ip.x:
                fingers_up += 1
        else:
            if thumb_tip.x > thumb_ip.x:
                fingers_up += 1

        # Other fingers: tip y < pip y means extended
        for tip_id, pip_id in zip(tip_ids[1:], pip_ids[1:]):
            if lms[tip_id].y < lms[pip_id].y:
                fingers_up += 1

        index_tip = lms[8]
        fingertip = (int(index_tip.x * w), int(index_tip.y * h))
        return max(0, min(5, fingers_up)), fingertip
