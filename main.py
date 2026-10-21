import os
import gc
import time
import uuid
import shutil
import tempfile
import subprocess
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import torch
import imageio_ffmpeg
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import supervision as sv

# ==============================================================================
# PyTorch Threading & Memory Optimization
# ==============================================================================
# Limit PyTorch CPU threads to avoid excessive thread memory overhead on Render Free Tier
torch.set_num_threads(min(2, os.cpu_count() or 1))

# ==============================================================================
# Directory Setup & App Initialization
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROCESSED_VIDEOS_DIR = os.path.join(STATIC_DIR, "processed_videos")
PROCESSED_IMAGES_DIR = os.path.join(STATIC_DIR, "processed_images")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)

app = FastAPI(
    title="SafeSite AI - Real-Time Computer Vision & Tracking Backend",
    version="2.2",
    description="Memory-optimized real-time personnel tracking, PPE safety compliance, HUD overlays, and web-streamable H.264 video encoding.",
)

# Enable CORS for Flutter mobile/web clients and local network devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mount for direct web/mobile streaming (natively supports GET, HEAD, and Range requests)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==============================================================================
# Model Loading (Primary Custom PPE Model ONLY)
# ==============================================================================
# Secondary yolov8n.pt general model has been disabled/removed to conserve RAM
# and stay strictly under Render Free Tier's 512MB RAM cap.
BEST_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[SafeSite Backend] Loading primary PPE model on device '{DEVICE}'...")

# Load ONLY the primary custom PPE model (contains all PPE classes and Person class)
if os.path.exists(BEST_MODEL_PATH):
    model_ppe = YOLO(BEST_MODEL_PATH)
    print(f"[SafeSite Backend] Loaded PPE model from '{BEST_MODEL_PATH}' ({len(model_ppe.names)} classes)")
else:
    model_ppe = YOLO("best.pt")
    print("[SafeSite Backend] Warning: best.pt not found, loading best.pt directly")

# Model aliases for convenience
model = model_ppe

# ------------------------------------------------------------------------------
# NOTE: Secondary general model (yolov8n.pt) removed to prevent OOM errors:
# YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
# model_person = YOLO(YOLO_MODEL_PATH)  # Disabled: reduces memory footprint by ~200MB
# ------------------------------------------------------------------------------

# Standard class names mapping from model_ppe
CLASS_NAMES: Dict[int, str] = model_ppe.names
# Find person class ID in model_ppe (class 5 'Person')
PERSON_CLASS_ID = 5
for cid, name in CLASS_NAMES.items():
    if name.lower() == "person":
        PERSON_CLASS_ID = cid
        break

# ==============================================================================
# Supervision Annotators
# ==============================================================================
box_annotator = sv.BoxAnnotator(
    thickness=2,
    color=sv.ColorPalette.DEFAULT,
)
corner_annotator = sv.BoxCornerAnnotator(
    thickness=2,
    corner_length=14,
    color=sv.ColorPalette.DEFAULT,
)
label_annotator = sv.LabelAnnotator(
    text_scale=0.48,
    text_thickness=1,
    text_padding=4,
    color=sv.ColorPalette.DEFAULT,
)

# ==============================================================================
# Helper Functions
# ==============================================================================
def create_session_tracker(fps: float = 25.0) -> sv.ByteTrack:
    """
    Creates an isolated ByteTrack tracker per video session to avoid cross-video state leaks.
    """
    return sv.ByteTrack(
        track_activation_threshold=0.20,
        lost_track_buffer=45,
        minimum_matching_threshold=0.75,
        frame_rate=max(1, int(fps)),
    )


def run_inference(
    frame: np.ndarray,
    conf: float = 0.20,
    imgsz: int = 320,
) -> sv.Detections:
    """
    Optimized single-model inference using primary PPE model (best.pt).
    Uses imgsz=320 (or 416) to significantly reduce CPU and RAM footprint during inference.
    """
    with torch.no_grad():
        results = model_ppe(frame, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)


def run_hybrid_inference(
    frame: np.ndarray,
    conf_ppe: float = 0.20,
    conf_person: float = 0.35,
    imgsz: int = 320,
) -> sv.Detections:
    """
    Backward-compatible wrapper routing to the memory-efficient single PPE model inference.
    """
    return run_inference(frame, conf=conf_ppe, imgsz=imgsz)


def is_person(class_id: int, class_name: str) -> bool:
    """Checks whether a detected class represents a human/person."""
    if class_id == PERSON_CLASS_ID:
        return True
    clean_name = class_name.lower().replace("-", "_").replace(" ", "_")
    return clean_name in ("person", "human", "worker")


def annotate_scene(
    scene: np.ndarray,
    detections: sv.Detections,
    labels: List[str],
) -> np.ndarray:
    """Applies box, corner brackets, and label annotations to a frame."""
    annotated = scene.copy()
    if len(detections) > 0:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = corner_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    return annotated


async def read_image_from_upload(file: UploadFile) -> Optional[np.ndarray]:
    """Reads and decodes an uploaded image file into a BGR numpy array."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def parse_detections(detections: sv.Detections):
    """Parses supervision Detections into structured objects, class counts, person count, and labels."""
    detected_objects_list: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    labels: List[str] = []
    people_count = 0

    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
        box = [round(float(coord), 1) for coord in detections.xyxy[i]]
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        if is_person(class_id, class_name):
            people_count += 1

        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        labels.append(f"{class_name} {int(confidence * 100)}%")

        detected_objects_list.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "box": box,
        })

    return detected_objects_list, class_counts, people_count, labels


def analyze_safety_compliance(detections_summary: Dict[str, int], people_count: int = 0) -> Dict[str, Any]:
    """
    Evaluates site safety compliance:
    - Identifies PPE gear (Hardhat, Safety Vest, Mask, Safety Cone, etc.)
    - Flags violations (NO-Hardhat, NO-Safety Vest, NO-Mask, etc.)
    - Calculates compliance score percentage (0.0 to 100.0)
    - Returns active violations dictionary and safety status (SAFE / WARNING / VIOLATION).
    """
    safe_count = 0
    violation_count = 0
    violations_detected: Dict[str, int] = {}

    for c_name, count in detections_summary.items():
        if count <= 0:
            continue
        clean_name = c_name.lower().replace("-", "_").replace(" ", "_")

        # Explicit negative PPE classes (e.g., NO-Hardhat, NO-Safety Vest, NO-Mask)
        if clean_name.startswith("no_") or "no_" in clean_name or "violation" in clean_name:
            violation_count += count
            violations_detected[c_name] = count
        elif clean_name in [
            "hardhat", "helmet", "safety_vest", "vest", "mask",
            "boots", "gloves", "safety_cone", "eye_protection",
            "head_protection", "foot_protection", "hand_protection"
        ]:
            safe_count += count

    total_checks = safe_count + violation_count

    if total_checks > 0:
        compliance_pct = round((safe_count / total_checks) * 100, 1)
        if violation_count == 0:
            safety_status = "SAFE"
        elif violation_count <= 2:
            safety_status = "WARNING"
        else:
            safety_status = "VIOLATION"
    else:
        # If no explicit PPE detected but people are present
        if people_count > 0:
            compliance_pct = 80.0
            safety_status = "SAFE"
        else:
            compliance_pct = 100.0
            safety_status = "SAFE"

    return {
        "compliance_score_percent": compliance_pct,
        "total_safe_items_detected": safe_count,
        "total_violations_detected": violation_count,
        "active_violations": violations_detected,
        "safety_status": safety_status,
    }


def draw_cctv_hud(
    frame: np.ndarray,
    current_people: int,
    peak_people: int,
    total_unique_people: int,
    frame_idx: int,
    total_frames: int = 0,
    safety_status: str = "SAFE",
) -> np.ndarray:
    """
    Renders a modern, resolution-adaptive glassmorphic CCTV HUD overlay on the video frame.
    """
    h, w = frame.shape[:2]
    scale = max(0.48, min(w / 1280.0, 1.3))

    pad = int(12 * scale)
    panel_w = int(410 * scale)
    panel_h = int(130 * scale)

    # Accent color based on status (BGR)
    if safety_status.upper() == "SAFE":
        status_color = (0, 220, 110)      # Green
    elif safety_status.upper() in ["WARNING", "ALERT"]:
        status_color = (0, 180, 255)     # Amber/Orange
    else:
        status_color = (60, 60, 240)      # Red

    # 1. Top-Left HUD Card
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (14, 18, 24), -1)
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), status_color, max(1, int(2 * scale)))
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    bold_thick = max(1, int(1.5 * scale))
    norm_thick = max(1, int(1.1 * scale))

    # Header
    cv2.putText(
        frame,
        "SAFESITE CCTV | REAL-TIME VISION",
        (pad + int(14 * scale), pad + int(25 * scale)),
        font,
        0.50 * scale,
        (210, 225, 235),
        norm_thick,
        cv2.LINE_AA,
    )
    # Live Occupancy
    cv2.putText(
        frame,
        f"LIVE PEOPLE: {current_people}",
        (pad + int(14 * scale), pad + int(54 * scale)),
        font,
        0.68 * scale,
        (255, 255, 255),
        bold_thick,
        cv2.LINE_AA,
    )
    # Peak & Total Unique
    cv2.putText(
        frame,
        f"PEAK: {peak_people}   |   UNIQUE TRACKED: {total_unique_people}",
        (pad + int(14 * scale), pad + int(82 * scale)),
        font,
        0.46 * scale,
        (185, 225, 255),
        norm_thick,
        cv2.LINE_AA,
    )
    # Status
    cv2.putText(
        frame,
        f"COMPLIANCE: {safety_status.upper()}",
        (pad + int(14 * scale), pad + int(112 * scale)),
        font,
        0.55 * scale,
        status_color,
        bold_thick,
        cv2.LINE_AA,
    )

    # 2. Bottom-Right Frame Counter Pill
    counter_w = int(210 * scale)
    counter_h = int(32 * scale)
    bx1 = w - pad - counter_w
    by1 = h - pad - counter_h
    overlay_br = frame.copy()
    cv2.rectangle(overlay_br, (bx1, by1), (bx1 + counter_w, by1 + counter_h), (14, 18, 24), -1)
    cv2.rectangle(overlay_br, (bx1, by1), (bx1 + counter_w, by1 + counter_h), (60, 75, 95), 1)
    cv2.addWeighted(overlay_br, 0.78, frame, 0.22, 0, frame)

    frame_text = f"FRAME: {frame_idx}/{total_frames}" if total_frames > 0 else f"FRAME: {frame_idx}"
    cv2.putText(
        frame,
        frame_text,
        (bx1 + int(10 * scale), by1 + int(21 * scale)),
        font,
        0.42 * scale,
        (220, 230, 240),
        1,
        cv2.LINE_AA,
    )

    return frame


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
def home():
    """System health check and loaded model diagnostics."""
    return {
        "status": "ok",
        "message": "SafeSite AI - Realtime CV Backend running!",
        "model_loaded": "best.pt (Optimized Single Model)",
        "device": DEVICE,
        "total_classes": len(CLASS_NAMES),
        "classes": list(CLASS_NAMES.values()),
        "endpoints": ["/detect-video/", "/detect/", "/detect-stream-frame/", "/video/{filename}", "/static/"],
    }


# Supports both GET and HEAD requests so Android ExoPlayer / VideoPlayer never receives 405
@app.api_route("/video/{filename}", methods=["GET", "HEAD"])
async def get_video(filename: str):
    """Direct video streaming endpoint supporting HTTP Range and HEAD requests."""
    file_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"status": "error", "message": "Video not found"})


@app.post("/detect/")
async def detect_objects(
    request: Request,
    file: UploadFile = File(..., description="Uploaded image file"),
    conf: float = Query(0.20, ge=0.05, le=0.95, description="Confidence threshold"),
    imgsz: int = Query(320, description="Inference resolution (e.g. 320 or 416)"),
):
    """Single image detection endpoint for PPE safety inspection."""
    img = None
    annotated_img = None
    try:
        img = await read_image_from_upload(file)
        if img is None:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid image format"})

        detections = run_inference(img, conf=conf, imgsz=imgsz)
        detected_objects_list, class_counts, people_count, labels = parse_detections(detections)

        # Annotate image
        annotated_img = annotate_scene(img, detections, labels)

        # Save annotated image
        img_id = uuid.uuid4().hex[:8]
        out_img_name = f"annotated_{img_id}.jpg"
        out_img_path = os.path.join(PROCESSED_IMAGES_DIR, out_img_name)
        cv2.imwrite(out_img_path, annotated_img)

        compliance_info = analyze_safety_compliance(class_counts, people_count=people_count)

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/static/processed_images/{out_img_name}"

        return {
            "status": "success",
            "total_objects_detected": len(detections),
            "people_count": people_count,
            "summary_counts": class_counts,
            "compliance": compliance_info,
            "detections": detected_objects_list,
            "annotated_image_url": image_url,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # Explicit memory cleanup
        del img, annotated_img
        gc.collect()


@app.post("/detect-stream-frame/")
async def detect_stream_frame(
    file: UploadFile = File(..., description="Camera frame image"),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold"),
    imgsz: int = Query(320, description="Inference resolution (e.g. 320 or 416)"),
):
    """Ultra-fast zero-disk I/O endpoint for live 6-15 FPS mobile camera inference."""
    img = None
    try:
        img = await read_image_from_upload(file)
        if img is None:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid frame"})

        h, w = img.shape[:2]
        t0 = time.time()
        detections = run_inference(img, conf=conf, imgsz=imgsz)
        inference_time_ms = round((time.time() - t0) * 1000, 1)

        detected_objects_list, class_counts, people_count, _ = parse_detections(detections)
        compliance_info = analyze_safety_compliance(class_counts, people_count=people_count)

        return {
            "status": "success",
            "image_width": w,
            "image_height": h,
            "inference_ms": inference_time_ms,
            "total_objects_detected": len(detections),
            "people_count": people_count,
            "summary_counts": class_counts,
            "compliance": compliance_info,
            "detections": detected_objects_list,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        del img


@app.post("/detect-video/")
async def detect_video(
    request: Request,
    file: UploadFile = File(..., description="Uploaded video file"),
    conf: float = Query(0.20, ge=0.05, le=0.95, description="Confidence threshold"),
    frame_stride: int = Query(2, ge=1, le=10, description="Frame stride optimization (e.g. 2 = every 2nd frame)"),
    imgsz: int = Query(320, description="Model inference resolution (e.g. 320 or 416)"),
):
    """
    Video File Upload & CCTV Bounding Bracket Tracking:
    1. Reads input video via streamed chunks (prevents RAM spikes).
    2. Applies frame stride and lightweight inference resolution (imgsz=320).
    3. Runs single PPE model inference and ByteTrack tracking.
    4. Annotates frames with bounding boxes, tracking IDs, confidence scores, and custom HUD overlays.
    5. Encodes into web-streamable H.264 MP4 format with yuv420p and faststart.
    6. Returns unique tracked people, peak concurrency, summary counts, compliance, and streamable URL.
    7. Cleans up all temp files and triggers immediate garbage collection in finally block.
    """
    temp_path: Optional[str] = None
    raw_out_path: Optional[str] = None
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None

    try:
        # 1. Save uploaded file to temp file using chunks to avoid loading full video into RAM
        file_ext = os.path.splitext(file.filename or "video.mp4")[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            while chunk := await file.read(1024 * 512):  # 512KB chunks
                temp.write(chunk)
            temp_path = temp.name

        # 2. Open VideoCapture
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {file.filename}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps) or fps > 60:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 0:
            total_frames = 0

        # Balance resolution for memory efficiency and ensure even dimensions for H.264 compliance
        out_width = min(frame_width, 640) if frame_width > 0 else 640
        out_height = int(frame_height * (out_width / frame_width)) if frame_width > 0 else 360
        out_width = (out_width // 2) * 2
        out_height = (out_height // 2) * 2

        video_id = str(uuid.uuid4())[:8]
        raw_out_path = os.path.join(tempfile.gettempdir(), f"raw_cctv_{video_id}.mp4")
        final_out_name = f"cctv_{video_id}.mp4"
        final_out_path = os.path.join(PROCESSED_VIDEOS_DIR, final_out_name)

        # Output FPS maintains 1:1 real-time playback duration with frame stride
        effective_fps = max(1.0, round(fps / frame_stride, 2))

        writer = cv2.VideoWriter(
            raw_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_fps,
            (out_width, out_height),
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to initialize OpenCV VideoWriter")

        # Isolated tracker instance for this video session
        tracker = create_session_tracker(fps=effective_fps)

        frame_count = 0
        processed_frames = 0
        max_people_in_frame = 0

        # Tracking state & statistics
        person_track_hits: Dict[int, int] = {}
        max_concurrent_classes: Dict[str, int] = {}

        peak_detections_list: List[Dict[str, Any]] = []
        latest_detections_list: List[Dict[str, Any]] = []

        # 3. Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_stride != 0:
                del frame
                continue

            processed_frames += 1

            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(frame, (out_width, out_height))

            # Run memory-efficient single PPE model inference (imgsz=320)
            detections = run_inference(frame, conf=conf, imgsz=imgsz)

            # Record concurrent detections in this frame
            frame_classes: Dict[str, int] = {}
            if len(detections) > 0:
                for cid in detections.class_id:
                    cname = CLASS_NAMES.get(int(cid), f"class_{cid}")
                    frame_classes[cname] = frame_classes.get(cname, 0) + 1

            for cname, count in frame_classes.items():
                max_concurrent_classes[cname] = max(max_concurrent_classes.get(cname, 0), count)

            # Update ByteTrack tracker
            tracked_detections = tracker.update_with_detections(detections)
            active_detections = tracked_detections if len(tracked_detections) > 0 else detections

            frame_people = 0
            current_frame_detections: List[Dict[str, Any]] = []
            labels: List[str] = []

            has_tracker_ids = (
                active_detections.tracker_id is not None
                and len(active_detections.tracker_id) == len(active_detections)
            )

            for i in range(len(active_detections)):
                cid = int(active_detections.class_id[i])
                cname = CLASS_NAMES.get(cid, f"class_{cid}")
                conf_val = float(active_detections.confidence[i]) if active_detections.confidence is not None else 0.0
                box_coords = [round(float(c), 1) for c in active_detections.xyxy[i]]

                tid: Optional[int] = None
                if has_tracker_ids and active_detections.tracker_id[i] is not None:
                    tid = int(active_detections.tracker_id[i])

                if is_person(cid, cname):
                    frame_people += 1
                    if tid is not None:
                        person_track_hits[tid] = person_track_hits.get(tid, 0) + 1

                # Formulate label
                if tid is not None:
                    label_text = f"#{tid} {cname} {int(conf_val * 100)}%"
                else:
                    label_text = f"{cname} {int(conf_val * 100)}%"
                labels.append(label_text)

                current_frame_detections.append({
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": round(conf_val, 2),
                    "box": box_coords,
                    "tracker_id": tid,
                })

            # Update peak people & detections
            if frame_people > max_people_in_frame:
                max_people_in_frame = frame_people
                peak_detections_list = current_frame_detections

            latest_detections_list = current_frame_detections

            # Annotate with Box, BoxCorner, and Label annotators
            annotated_frame = annotate_scene(frame, active_detections, labels)

            # Determine provisional status for live HUD
            live_status = "SAFE"
            if any(k.lower().startswith("no_") or "no_" in k.lower() for k in frame_classes.keys()):
                live_status = "VIOLATION"

            # Dynamic people count for HUD
            hud_people_display = max(frame_people, max_people_in_frame)

            # Render custom glassmorphic CCTV HUD overlay
            annotated_frame = draw_cctv_hud(
                frame=annotated_frame,
                current_people=frame_people,
                peak_people=max_people_in_frame,
                total_unique_people=hud_people_display,
                frame_idx=processed_frames,
                total_frames=max(1, total_frames // frame_stride) if total_frames > 0 else processed_frames,
                safety_status=live_status,
            )

            writer.write(annotated_frame)

            # Explicitly free per-frame allocated memory
            del frame, annotated_frame, detections, tracked_detections, active_detections

            # Periodically collect garbage on longer video clips
            if processed_frames % 25 == 0:
                gc.collect()

        # Release capture & writer before transcoding to unlock files
        if cap is not None:
            cap.release()
            cap = None
        if writer is not None:
            writer.release()
            writer = None

        # 4. Transcode to web-streamable H.264 MP4 using imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe, "-y",
            "-i", raw_out_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            final_out_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 5. Robust People & Object Counting (removes phantom IDs caused by track switching)
        confirmed_person_tracks = [tid for tid, hits in person_track_hits.items() if hits >= 3]

        if max_people_in_frame > 0:
            max_plausible_people = max(max_people_in_frame, int(max_people_in_frame * 1.5) + 1)
            if len(confirmed_person_tracks) > 0:
                unique_people = min(len(confirmed_person_tracks), max_plausible_people)
                unique_people = max(unique_people, max_people_in_frame)
            else:
                unique_people = max_people_in_frame
        else:
            unique_people = len(confirmed_person_tracks)

        final_summary: Dict[str, int] = {}
        for cname, count in max_concurrent_classes.items():
            if count > 0:
                final_summary[cname] = count

        if unique_people > 0:
            final_summary["Person"] = unique_people

        compliance_info = analyze_safety_compliance(final_summary, people_count=unique_people)

        # Use /static/processed_videos/ path so ExoPlayer always receives 200/206 with Range support
        base_url = str(request.base_url).rstrip("/")
        video_full_url = f"{base_url}/static/processed_videos/{final_out_name}"

        rep_detections = peak_detections_list if peak_detections_list else latest_detections_list

        return {
            "status": "success",
            "total_frames_processed": processed_frames,
            "raw_frames_read": frame_count,
            "people_count": unique_people,
            "unique_objects_tracked": unique_people,
            "peak_concurrent_people": max_people_in_frame,
            "total_objects_detected": sum(final_summary.values()),
            "summary_counts": final_summary,
            "compliance": compliance_info,
            "annotated_video_url": video_full_url,
            "video_filename": final_out_name,
            "detections": rep_detections,
        }

    finally:
        # Crucial cleanup in finally block: release file handles, delete temp files, and garbage collect
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        if raw_out_path and os.path.exists(raw_out_path):
            try:
                os.unlink(raw_out_path)
            except Exception:
                pass

        # Force garbage collection to instantly free unreferenced RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
