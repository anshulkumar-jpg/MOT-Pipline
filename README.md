# Identity-Aware Multi-Object Tracking System

Reference implementation of the pipeline described in the accompanying
write-up: a multi-branch person Re-Identification (ReID) model trained
jointly with attribute prediction, an automatic attribute-annotation
pipeline, and a tracker that gates candidate detections using motion and
trajectory reasoning *before* running ReID, rather than relying on raw
IoU overlap alone.

## Layout

```
identity_mot/
├── detector/          # object detection backends (torchvision / precomputed)
├── reid/              # multi-branch ReID model, losses, backbone
├── tracking/          # Kalman motion model, trajectory features,
│                      # motion/geometry candidate gating, ReID-aware
│                      # association, track lifecycle management
├── annotation/         # automatic attribute pseudo-labeling + dataset builder
├── evaluation/         # MOT17 / MOT20 evaluation drivers + metrics (motmetrics)
├── configs/mot.yaml    # all pipeline parameters in one place
└── main.py             # CLI: track / annotate / evaluate
```

## Pipeline

```
Automatic Data Annotation
          |
Multi-Branch ReID (identity + attribute branches, joint loss)
          |
Object Detection
          |
Motion & Trajectory Reasoning (Kalman filter, velocity, direction,
          |                     aspect-ratio change)
Spatial Candidate Gating (Mahalanobis-gated, not raw IoU)
          |
Historical ReID Matching (track embedding gallery, not single-frame)
          |
Hungarian Association -> Final Object Identity
```

## Quick start

```bash
pip install -r requirements.txt

# Run tracker on a video and save the final output video with bounding boxes & track IDs
python main.py track --video path/to/video.mp4 --save-video results/output.mp4

# Run tracker and save MOTChallenge text results
python main.py track --video path/to/video.mp4 --out results/track.txt

# Run tracker saving both output video and text results with live visualization
python main.py track --video path/to/video.mp4 --save-video results/output.mp4 --out results/track.txt --visualize

# Build an auto-annotated attribute dataset from tracked crops
python main.py annotate --frames-dir path/to/frames --tracks path/to/tracks.json

# Evaluate on MOT17 and MOT20
python main.py evaluate --benchmark both --dataset-root data/MOT17
```

All hyperparameters (gating thresholds, association weights, loss
weights, training schedule) live in `configs/mot.yaml`.

## Running on CPU only

The default config (`configs/mot.yaml`) is already CPU-safe:
`detector.device` and `reid.device` are both set to `cpu`, so nothing
in the pipeline probes for or requires a GPU. To be explicit or to
enable a GPU when one is present:

- `device: cpu` — always CPU, never touches `torch.cuda`.
- `device: cuda` — always GPU (fails if none is available).
- `device: auto` — GPU if `torch.cuda.is_available()`, else CPU.

For CPU-only inference/tracking, install the CPU build of torch (see
the note at the top of `requirements.txt`) — it's a much smaller
download with no CUDA runtime, and every module here (Kalman motion
model, gating, association, ReID inference) runs correctly on it.
Training the ReID backbone from scratch on CPU is possible but slow;
see the comment in `configs/mot.yaml`'s `training:` section for
CPU-appropriate batch-size settings.

## Key design points

- **`tracking/candidate_gating.py`**: replaces "match the highest-IoU
  box" with "which detections are physically/temporally plausible given
  this track's velocity, direction, trajectory, and aspect-ratio
  history?" — ReID is only run against the surviving candidates.
- **`reid/losses.py`**: `JointReIDAttributeLoss` trains the identity and
  attribute branches together so the shared backbone learns a richer,
  more robust representation than identity labels alone provide.
- **`annotation/auto_attribute.py`**: generates attribute pseudo-labels
  at scale (with confidence gating and per-track majority-vote
  smoothing) so the joint loss above doesn't require exhaustive manual
  attribute annotation.
- **`tracking/track.py`**: keeps a small embedding *gallery* per track
  (not just the last embedding), so identity matching is robust to a
  single bad-appearance frame.
