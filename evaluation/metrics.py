"""
metrics.py
----------
Standard MOT metrics (MOTA, MOTP, IDF1, ID switches, mostly-tracked /
mostly-lost, etc.), computed with `motmetrics` against MOTChallenge-format
ground-truth and result files. Shared by evaluation/mot17.py and
evaluation/mot20.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import motmetrics as mm
    _HAS_MOTMETRICS = True
except ImportError:  # pragma: no cover
    _HAS_MOTMETRICS = False


@dataclass
class SequenceResult:
    sequence_name: str
    metrics: Dict[str, float]


def load_motchallenge_file(path: str) -> np.ndarray:
    """
    Loads a MOTChallenge-format file:
        frame, id, x, y, w, h, score/conf, class, visibility
    Returns raw (N, >=6) array.
    """
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    return data


def _to_motmetrics_frame(rows: np.ndarray):
    """Groups raw MOTChallenge rows by frame -> {frame_id: (ids, boxes)}."""
    by_frame: Dict[int, tuple] = {}
    for row in rows:
        frame_id = int(row[0])
        obj_id = int(row[1])
        x, y, w, h = row[2:6]
        by_frame.setdefault(frame_id, {"ids": [], "boxes": []})
        by_frame[frame_id]["ids"].append(obj_id)
        by_frame[frame_id]["boxes"].append([x, y, w, h])
    return by_frame


def compute_sequence_metrics(
    gt_path: str,
    result_path: str,
    sequence_name: str = "seq",
    iou_threshold: float = 0.5,
) -> SequenceResult:
    """
    Computes the standard metric set for one sequence given a ground
    truth file and a tracker-result file, both in MOTChallenge format.
    """
    if not _HAS_MOTMETRICS:
        raise RuntimeError(
            "The 'motmetrics' package is required for evaluation. "
            "Install with `pip install motmetrics`."
        )

    gt_rows = load_motchallenge_file(gt_path)
    res_rows = load_motchallenge_file(result_path)

    gt_by_frame = _to_motmetrics_frame(gt_rows)
    res_by_frame = _to_motmetrics_frame(res_rows)

    accumulator = mm.MOTAccumulator(auto_id=False)

    all_frames = sorted(set(gt_by_frame.keys()) | set(res_by_frame.keys()))
    for frame_id in all_frames:
        gt = gt_by_frame.get(frame_id, {"ids": [], "boxes": []})
        res = res_by_frame.get(frame_id, {"ids": [], "boxes": []})

        distances = mm.distances.iou_matrix(
            np.array(gt["boxes"]) if gt["boxes"] else np.empty((0, 4)),
            np.array(res["boxes"]) if res["boxes"] else np.empty((0, 4)),
            max_iou=1 - iou_threshold,
        )
        accumulator.update(gt["ids"], res["ids"], distances, frameid=frame_id)

    metrics_host = mm.metrics.create()
    summary = metrics_host.compute(
        accumulator,
        metrics=[
            "mota", "motp", "idf1", "idp", "idr",
            "num_switches", "num_false_positives", "num_misses",
            "mostly_tracked", "mostly_lost", "num_fragmentations",
        ],
        name=sequence_name,
    )

    result_dict = {col: float(summary.iloc[0][col]) for col in summary.columns}
    return SequenceResult(sequence_name=sequence_name, metrics=result_dict)


def aggregate_metrics(results: list) -> Dict[str, float]:
    """Simple unweighted mean across sequences (a weighted-by-length
    aggregate can be substituted if per-sequence frame counts are tracked)."""
    if not results:
        return {}
    keys = results[0].metrics.keys()
    return {
        key: float(np.mean([r.metrics[key] for r in results]))
        for key in keys
    }


def format_report(results: list, aggregate: Optional[Dict[str, float]] = None) -> str:
    lines = [f"{'Sequence':<20}{'MOTA':>8}{'IDF1':>8}{'IDSw':>8}{'MT':>6}{'ML':>6}"]
    for r in results:
        m = r.metrics
        lines.append(
            f"{r.sequence_name:<20}{m.get('mota', 0):>8.3f}{m.get('idf1', 0):>8.3f}"
            f"{int(m.get('num_switches', 0)):>8}{int(m.get('mostly_tracked', 0)):>6}"
            f"{int(m.get('mostly_lost', 0)):>6}"
        )
    if aggregate:
        lines.append("-" * 56)
        lines.append(
            f"{'OVERALL':<20}{aggregate.get('mota', 0):>8.3f}{aggregate.get('idf1', 0):>8.3f}"
        )
    return "\n".join(lines)
