"""Canonical AprilCube correspondence records and hashes."""

from __future__ import annotations

import hashlib
import json

from aprilcube import CorrespondenceResult


def correspondence_to_dict(result: CorrespondenceResult) -> dict:
    return {
        "image_size_wh": list(result.image_size_wh),
        "observations": [
            {
                "tag_id": item.tag_id,
                "face_name": item.face_name,
                "image_corners_px": item.image_corners_px.tolist(),
                "object_corners_mm": item.object_corners_mm.tolist(),
                "quad_quality": item.quad_quality,
                "shortest_side_px": item.shortest_side_px,
                "image_margin_px": item.image_margin_px,
            }
            for item in result.observations
        ],
        "duplicate_tag_ids": list(result.duplicate_tag_ids),
        "ignored_tag_ids": list(result.ignored_tag_ids),
        "opencv_rejected_candidates": result.opencv_rejected_candidates,
        "quality_rejected_detections": result.quality_rejected_detections,
    }


def correspondence_sha256(result: CorrespondenceResult) -> str:
    encoded = json.dumps(
        correspondence_to_dict(result),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
