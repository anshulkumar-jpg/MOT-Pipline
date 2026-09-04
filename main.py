"""
main.py
-------
Entry point for the Identity-Aware Multi-Object Tracking pipeline.

Supports three modes:

  track   -- run detection + multi-branch ReID + motion/geometry-gated
             association on a video or image sequence, writing a
             MOTChallenge-format result file (and optionally an
             annotated video).
  annotate -- run the automatic attribute-annotation pipeline over a
             folder of tracked crops to build a training dataset
             (annotation/auto_attribute.py + dataset_builder.py).
  evaluate -- run the full tracker over MOT17 and/or MOT20 and report
             MOTA / IDF1 / ID-switch metrics (evaluation/mot17.py,
             evaluation/mot20.py).

Usage:
    python main.py track --video path/to/video.mp4 --out results/track.txt
    python main.py annotate --frames-dir path/to/frames --tracks path/to/tracks.json
    python main.py evaluate --benchmark mot17 --dataset-root data/MOT17
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _format_time(seconds: float) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_gstreamer_input_pipeline(source: str) -> str:
    source_str = str(source).strip()
    if "!" in source_str:
        return source_str
    if source_str.startswith("rtsp://"):
        return f"rtspsrc location={source_str} latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    if source_str.startswith("udp://"):
        port = source_str.split(":")[-1].replace("/", "")
        return f"udpsrc port={port} caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264\" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    return f"filesrc location=\"{source_str}\" ! decodebin ! videoconvert ! video/x-raw, format=BGR ! appsink"


# ---------------------------------------------------------------------- #
# track: run the live pipeline over a video / image sequence
# ---------------------------------------------------------------------- #
def run_track(args: argparse.Namespace, config: dict) -> None:
    import cv2
    import torch

    try:
        torch.set_num_threads(max(4, os.cpu_count() or 4))
    except Exception:
        pass

    from detector.detector import TorchvisionDetector, YOLODetector
    from reid.embedding import MultiBranchReID, preprocess_crop
    from tracking.track_manager import TrackManager, TrackManagerConfig
    from tracking.candidate_gating import GatingConfig
    from tracking.association import AssociationConfig

    det_cfg = config["detector"]
    reid_cfg = config["reid"]

    score_thresh = getattr(args, "score_threshold", None)
    if score_thresh is None:
        score_thresh = getattr(args, "confidence", None)
    if score_thresh is None:
        score_thresh = det_cfg.get("score_threshold", 0.25)

    backend = (getattr(args, "backend", None) or det_cfg.get("backend", "volo26n")).lower()
    device = getattr(args, "device", None) or det_cfg.get("device", "auto")

    cli_imgsz = getattr(args, "imgsz", None)
    cli_tiling = getattr(args, "tiling", None)

    if getattr(args, "fast", False):
        imgsz = cli_imgsz if cli_imgsz is not None else 640
        tiling = cli_tiling if cli_tiling is not None else False
    else:
        imgsz = cli_imgsz if cli_imgsz is not None else det_cfg.get("imgsz", 352)
        tiling = cli_tiling if cli_tiling is not None else det_cfg.get("tiling", True)

    if backend in ("volo26n", "yolo26n", "volo26m", "yolo26m", "yolo", "volo", "ultralytics", "yolov8") or backend.startswith("yolo") or backend.startswith("volo"):
        model_name = getattr(args, "model_name", None) or det_cfg.get("model_name", "volo26n.pt")
        detector = YOLODetector(
            model_name=model_name,
            weights_path=det_cfg.get("weights_path"),
            score_threshold=score_thresh,
            imgsz=imgsz,
            tiling=tiling,
            grid_split_3x3=det_cfg.get("grid_split_3x3", True),
            include_full_frame=det_cfg.get("include_full_frame", False),
            tile_size=det_cfg.get("tile_size", 640),
            tile_overlap=det_cfg.get("tile_overlap", 0.15),
            device=device,
        )
    else:
        detector = TorchvisionDetector(
            weights_path=det_cfg.get("weights_path"),
            score_threshold=score_thresh,
            device=device,
        )

    reid_device = getattr(args, "device", None) or reid_cfg.get("device", "auto")
    reid_model = MultiBranchReID(
        num_identities=reid_cfg.get("num_identities", 0),
        embed_dim=reid_cfg.get("embed_dim", 512),
        pretrained_backbone=reid_cfg.get("pretrained_backbone", True),
        device=reid_device,
    )
    if reid_cfg.get("checkpoint"):
        # map_location="cpu" is always safe here, whether the checkpoint was
        # trained on GPU or CPU -- load_state_dict then copies the values
        # into reid_model's parameters on whatever device it actually lives on.
        state = torch.load(reid_cfg["checkpoint"], map_location="cpu")
        reid_model.load_state_dict(state, strict=False)
    reid_model.eval()

    tm_cfg = config.get("track_manager", {}).copy()
    if score_thresh is not None:
        tm_cfg["min_detection_score"] = score_thresh

    manager = TrackManager(
        TrackManagerConfig(**tm_cfg),
        gating_config=GatingConfig(**config.get("gating", {})),
        association_config=AssociationConfig(**config.get("association", {})),
    )

    gst_cfg = config.get("gstreamer", {})
    use_gstreamer = (
        getattr(args, "gstreamer", False)
        or gst_cfg.get("enabled", False)
        or args.video.startswith(("rtsp://", "udp://"))
        or "!" in args.video
    )

    if use_gstreamer:
        input_gst = gst_cfg.get("input_pipeline") or _build_gstreamer_input_pipeline(args.video)
        print(f"[GStreamer] Opening input pipeline:\n  {input_gst}")
        cap = cv2.VideoCapture(input_gst, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("[GStreamer Info] Note: Standard PyPI `opencv-python` on Windows is built without GStreamer C++ bindings.")
            clean_source = args.video
            if "location=" in clean_source:
                try:
                    clean_source = clean_source.split("location=")[1].split("!")[0].strip("\"' ")
                except Exception:
                    pass
            print(f"[GStreamer Info] Falling back to direct video reader: {clean_source}")
            cap = cv2.VideoCapture(clean_source)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = getattr(args, "max_frames", None)

    if max_frames is not None and max_frames > 0:
        target_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
    else:
        target_frames = total_frames

    video_len_sec = total_frames / video_fps if total_frames > 0 and video_fps > 0 else 0

    print(f"\n[MOT Tracker] Video: {args.video}")
    if target_frames > 0:
        print(f"[MOT Tracker] Total Frames to process: {target_frames} | Video Length: {_format_time(video_len_sec)} @ {video_fps:.1f} FPS")
    print("[MOT Tracker] Starting tracking pipeline...\n")

    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        if use_gstreamer:
            output_gst = gst_cfg.get("output_pipeline") or (
                f"appsrc ! videoconvert ! video/x-raw, format=BGR ! "
                f"x264enc speed-preset=ultrafast tune=zerolatency ! mp4mux ! "
                f"filesink location=\"{args.save_video}\""
            )
            print(f"[GStreamer] Opening output pipeline:\n  {output_gst}")
            writer = cv2.VideoWriter(output_gst, cv2.CAP_GSTREAMER, 0, video_fps, (w, h))
            if not writer.isOpened():
                print("[GStreamer Warning] Could not open GStreamer VideoWriter, using OpenCV mp4v fallback...")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, video_fps, (w, h))
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.save_video, fourcc, video_fps, (w, h))

    result_rows = []
    frame_id = 0
    start_time = time.time()
    seen_track_ids = set()

    while True:
        if max_frames is not None and frame_id >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        detections = detector.detect(frame)
        if detections:
            boxes = np.stack([d.box for d in detections])
            scores = np.array([d.score for d in detections], dtype=np.float32)
            crops = []
            for det in detections:
                x1, y1, x2, y2 = det.box.astype(int)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    crop = np.zeros((160, 80, 3), dtype=np.uint8)
                crops.append(preprocess_crop(crop))
            batch = torch.stack(crops)
            embeddings = reid_model.embed(batch)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            embeddings = np.empty((0, reid_model.embed_dim), dtype=np.float32)

        confirmed_tracks = manager.step(boxes, scores, embeddings)

        for trk in confirmed_tracks:
            seen_track_ids.add(trk.track_id)

        for row in manager.results_as_mot_rows():
            result_rows.append(row)

        if args.save_video or args.visualize:
            _draw_tracks(frame, confirmed_tracks, total_unique_ids=len(seen_track_ids))

        if writer is not None:
            writer.write(frame)

        if args.visualize:
            cv2.imshow("Identity-Aware MOT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Terminal progress reporting with ETA (Estimated Time Remaining)
        elapsed = time.time() - start_time
        fps_speed = frame_id / elapsed if elapsed > 0 else 0.0

        if target_frames > 0:
            pct = min(100.0, (frame_id / target_frames) * 100)
            remaining_frames = max(0, target_frames - frame_id)
            eta_sec = remaining_frames / fps_speed if fps_speed > 0 else 0.0

            bar_len = 20
            filled_len = int(bar_len * frame_id // target_frames)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)

            sys.stdout.write(
                f"\r[Progress] [{bar}] {pct:5.1f}% | Frame {frame_id}/{target_frames} | "
                f"Speed: {fps_speed:4.1f} fps | Elapsed: {_format_time(elapsed)} | ETA (Remaining Time): {_format_time(eta_sec)}  "
            )
            sys.stdout.flush()
        else:
            sys.stdout.write(
                f"\r[Progress] Frame {frame_id} | Speed: {fps_speed:4.1f} fps | Elapsed: {_format_time(elapsed)}  "
            )
            sys.stdout.flush()

    if frame_id > 0:
        sys.stdout.write("\n")
        total_elapsed = time.time() - start_time
        avg_fps = frame_id / total_elapsed if total_elapsed > 0 else 0.0
        print(f"\n[MOT Tracker] Tracking complete! Processed {frame_id} frames in {_format_time(total_elapsed)} (Avg Speed: {avg_fps:.1f} fps)")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video to {args.save_video}")
    if args.visualize:
        cv2.destroyAllWindows()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        _write_mot_result(result_rows, args.out)
        print(f"Wrote {len(result_rows)} track rows across {frame_id} frames to {args.out}")


def _draw_tracks(frame, tracks, total_unique_ids: Optional[int] = None) -> None:
    import cv2
    colors = [
        (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
        (207, 210, 49), (72, 249, 10), (14, 249, 152), (148, 241, 248),
        (47, 161, 255), (155, 47, 255), (255, 47, 226), (235, 100, 150)
    ]
    for track in tracks:
        x1, y1, x2, y2 = track.box_xyxy.astype(int)
        color = colors[track.track_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 6, max(0, y1)), color, -1)
        cv2.putText(frame, label, (x1 + 3, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Draw Person Count Badge in Top-Left Corner (Large & Prominent) ---
    h, w = frame.shape[:2]
    scale = max(1.15, w / 1100.0)
    thickness = max(2, int(scale * 2.2))

    live_count = len(tracks)
    text = f"Total Persons: {live_count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    margin = int(20 * (scale / 1.2))
    pad_x, pad_y = int(18 * scale), int(14 * scale)
    x_min, y_min = margin, margin
    x_max, y_max = margin + text_w + (pad_x * 2), margin + text_h + (pad_y * 2)

    # Draw semi-transparent dark background card with thick cyan accent border
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (15, 15, 15), -1)
    border_thick = max(2, int(scale * 2.2))
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 215, 255), border_thick)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Render bold white text
    text_x = x_min + pad_x
    text_y = y_min + text_h + pad_y - 2
    cv2.putText(frame, text, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _write_mot_result(rows, path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(",".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + "\n")


# ---------------------------------------------------------------------- #
# annotate: run the automatic attribute annotation pipeline
# ---------------------------------------------------------------------- #
def run_annotate(args: argparse.Namespace, config: dict) -> None:
    import json
    import cv2
    import torch

    from reid.embedding import MultiBranchReID, preprocess_crop
    from annotation.auto_attribute import AutoAttributeAnnotator, AttributeAnnotationConfig
    from annotation.dataset_builder import DatasetBuilder

    reid_cfg = config["reid"]
    reid_device = reid_cfg.get("device", "cpu")
    reid_model = MultiBranchReID(
        num_identities=0,
        embed_dim=reid_cfg.get("embed_dim", 512),
        pretrained_backbone=True,
        device=reid_device,
    )
    if reid_cfg.get("checkpoint"):
        state = torch.load(reid_cfg["checkpoint"], map_location="cpu")
        reid_model.load_state_dict(state, strict=False)
    reid_model.eval()

    annotator = AutoAttributeAnnotator(
        attribute_model=reid_model,
        config=AttributeAnnotationConfig(**config.get("annotation", {})),
        device=reid_device,
    )

    with open(args.tracks, "r") as f:
        track_data = json.load(f)  # {frame_id: [{track_id, box}, ...]}

    frame_files = sorted(os.listdir(args.frames_dir))
    frames = [cv2.imread(os.path.join(args.frames_dir, fname)) for fname in frame_files]

    frame_track_boxes = []
    for frame_id in range(len(frames)):
        entries = track_data.get(str(frame_id), [])
        frame_track_boxes.append([
            (entry["track_id"], np.array(entry["box"], dtype=np.float32))
            for entry in entries
        ])

    builder = DatasetBuilder(
        dataset_root=config["annotation_pipeline"]["dataset_root"],
        val_ratio=config["annotation_pipeline"].get("val_ratio", 0.1),
        seed=config["annotation_pipeline"].get("seed", 42),
    )

    # First pass: build crops without attribute labels, to get a stable ordering.
    builder.build_from_tracks(
        frames, frame_track_boxes, attribute_labels=None,
        padding=config["annotation_pipeline"].get("crop_padding", 0.1),
    )

    crops_batch = []
    track_ids_flat = []
    for _, rec in builder._records:
        crops_batch.append(preprocess_crop(rec.crop))
        track_ids_flat.append(rec.track_id)

    batch_tensor = torch.stack(crops_batch)
    labels = annotator.annotate_batch(batch_tensor)
    labels = annotator._apply_track_consistency(labels, np.array(track_ids_flat)) \
        if annotator.config.enforce_track_consistency else labels

    for i, (_, rec) in enumerate(builder._records):
        rec.attributes = {name: int(values[i]) for name, values in labels.items()}

    csv_path = builder.write_annotations_csv()
    train_path, val_path = builder.write_splits()
    print(f"Wrote dataset annotations to {csv_path}")
    print(f"Train split: {train_path}, Val split: {val_path}")


# ---------------------------------------------------------------------- #
# evaluate: run on MOT17 / MOT20 and report metrics
# ---------------------------------------------------------------------- #
def run_evaluate(args: argparse.Namespace, config: dict) -> None:
    import torch
    from reid.embedding import MultiBranchReID

    reid_cfg = config["reid"]
    reid_model: Optional[MultiBranchReID] = None
    if reid_cfg.get("checkpoint"):
        reid_model = MultiBranchReID(
            num_identities=0,
            embed_dim=reid_cfg.get("embed_dim", 512),
            pretrained_backbone=False,
            device=reid_cfg.get("device", "cpu"),
        )
        state = torch.load(reid_cfg["checkpoint"], map_location="cpu")
        reid_model.load_state_dict(state, strict=False)
        reid_model.eval()

    if args.benchmark in ("mot17", "both"):
        from evaluation.mot17 import MOT17Evaluator, MOT17EvalConfig
        eval_cfg_dict = dict(config["evaluation"]["mot17"])
        if args.dataset_root:
            eval_cfg_dict["dataset_root"] = args.dataset_root
        eval_cfg = MOT17EvalConfig(**eval_cfg_dict)
        report = MOT17Evaluator(eval_cfg, reid_model=reid_model).evaluate()
        print("=== MOT17 ===")
        print(report or "(no ground truth found -- ran inference only)")

    if args.benchmark in ("mot20", "both"):
        from evaluation.mot20 import MOT20Evaluator, MOT20EvalConfig
        from tracking.candidate_gating import GatingConfig
        from tracking.association import AssociationConfig
        eval_cfg_dict = dict(config["evaluation"]["mot20"])
        gating_dict = eval_cfg_dict.pop("gating", {})
        assoc_dict = eval_cfg_dict.pop("association", {})
        if args.dataset_root:
            eval_cfg_dict["dataset_root"] = args.dataset_root
        eval_cfg = MOT20EvalConfig(
            gating_config=GatingConfig(**gating_dict) if gating_dict else GatingConfig(max_center_distance=100.0),
            association_config=AssociationConfig(**assoc_dict) if assoc_dict else AssociationConfig(reid_weight=0.8, motion_weight=0.2, max_reid_distance=0.4),
            **eval_cfg_dict
        )
        report = MOT20Evaluator(eval_cfg, reid_model=reid_model).evaluate()
        print("=== MOT20 ===")
        print(report or "(no ground truth found -- ran inference only)")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identity-Aware Multi-Object Tracking pipeline")
    parser.add_argument("--config", default="configs/mot.yaml", help="Path to YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    track_parser = subparsers.add_parser("track", help="Run tracking on a video")
    track_parser.add_argument("--video", required=True, help="Path to input video file")
    track_parser.add_argument("--out", default=None, help="Output MOTChallenge-format result file (.txt, optional)")
    track_parser.add_argument("--save-video", default=None, help="Path to save output annotated video (.mp4/.avi, optional)")
    track_parser.add_argument("--visualize", action="store_true", help="Display live tracking window")
    track_parser.add_argument("--score-threshold", "--confidence", type=float, default=None, help="Detection score / confidence threshold override")
    track_parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    track_parser.add_argument("--backend", default=None, help="Detector backend (volo26n, torchvision, etc.)")
    track_parser.add_argument("--model-name", default=None, help="Detector model name/weights path (e.g. volo26n.pt)")
    track_parser.add_argument("--imgsz", type=int, default=None, help="Inference resolution size (e.g. 640 for fast CPU, 1280 for high precision)")
    track_parser.add_argument("--tiling", action="store_true", default=None, help="Enable multi-pass grid tiling for small/dense crowd detections (slower)")
    track_parser.add_argument("--no-tiling", action="store_false", dest="tiling", help="Disable multi-pass grid tiling for 10x-15x faster speed")
    track_parser.add_argument("--device", default=None, help="Device to run inference on: 'cpu', 'cuda', or 'auto'")
    track_parser.add_argument("--fast", action="store_true", help="Enable fast mode (sets imgsz=640 and disables tiling for 10x-15x speedup)")
    track_parser.add_argument("--gstreamer", "--use-gstreamer", action="store_true", help="Enable GStreamer backend pipeline for RTSP / UDP / file video streams")

    annotate_parser = subparsers.add_parser("annotate", help="Run automatic attribute annotation")
    annotate_parser.add_argument("--frames-dir", required=True)
    annotate_parser.add_argument("--tracks", required=True, help="JSON file: {frame_id: [{track_id, box}]}")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on MOT17/MOT20")
    eval_parser.add_argument("--benchmark", choices=["mot17", "mot20", "both"], default="both")
    eval_parser.add_argument("--dataset-root", default=None)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "track":
        run_track(args, config)
    elif args.command == "annotate":
        run_annotate(args, config)
    elif args.command == "evaluate":
        run_evaluate(args, config)


if __name__ == "__main__":
    main()

