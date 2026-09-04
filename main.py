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
import os
from typing import Optional

import numpy as np
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------- #
# track: run the live pipeline over a video / image sequence
# ---------------------------------------------------------------------- #
def run_track(args: argparse.Namespace, config: dict) -> None:
    import cv2
    import torch

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

    backend = (getattr(args, "backend", None) or det_cfg.get("backend", "yolo26m")).lower()
    if backend in ("yolo26m", "volo26m", "yolo", "ultralytics", "yolov8") or backend.startswith("yolo") or backend.startswith("volo"):
        model_name = getattr(args, "model_name", None) or det_cfg.get("model_name", "yolo26m.pt")
        imgsz = getattr(args, "imgsz", None) or det_cfg.get("imgsz", 1280)
        tiling = getattr(args, "tiling", None)
        if tiling is None:
            tiling = det_cfg.get("tiling", True)
        detector = YOLODetector(
            model_name=model_name,
            weights_path=det_cfg.get("weights_path"),
            score_threshold=score_thresh,
            imgsz=imgsz,
            tiling=tiling,
            tile_size=det_cfg.get("tile_size", 640),
            tile_overlap=det_cfg.get("tile_overlap", 0.2),
            device=det_cfg.get("device"),
        )
    else:
        detector = TorchvisionDetector(
            weights_path=det_cfg.get("weights_path"),
            score_threshold=score_thresh,
            device=det_cfg.get("device"),
        )

    reid_model = MultiBranchReID(
        num_identities=reid_cfg.get("num_identities", 0),
        embed_dim=reid_cfg.get("embed_dim", 512),
        pretrained_backbone=reid_cfg.get("pretrained_backbone", True),
        device=reid_cfg.get("device", "cpu"),
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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))

    result_rows = []
    frame_id = 0
    max_frames = getattr(args, "max_frames", None)
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
                    crop = np.zeros((256, 128, 3), dtype=np.uint8)
                crops.append(preprocess_crop(crop))
            batch = torch.stack(crops)
            embeddings = reid_model.embed(batch)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            embeddings = np.empty((0, reid_model.embed_dim), dtype=np.float32)

        confirmed_tracks = manager.step(boxes, scores, embeddings)

        for row in manager.results_as_mot_rows():
            result_rows.append(row)

        if args.save_video or args.visualize:
            _draw_tracks(frame, confirmed_tracks)

        if writer is not None:
            writer.write(frame)

        if args.visualize:
            cv2.imshow("Identity-Aware MOT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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


def _draw_tracks(frame, tracks) -> None:
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
    track_parser.add_argument("--backend", default=None, help="Detector backend (yolo26m, torchvision, etc.)")
    track_parser.add_argument("--model-name", default=None, help="Detector model name/weights path (e.g. yolo26m.pt)")

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

