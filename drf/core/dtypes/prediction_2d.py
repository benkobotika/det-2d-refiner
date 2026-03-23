from dataclasses import dataclass, field
from typing import List, Dict, Any

from drf.core.dtypes.image_metadata import ImageMetadata


@dataclass
class Prediction2D:
    bboxes: List[List[float]] = field(
        default_factory=list
    )  # List of bounding boxes, each represented as [x_top_left, y_top_left, width, height]
    labels: List[str] = field(
        default_factory=list
    )  # List of class labels corresponding to each bounding box
    bbox_scores: List[float] = field(
        default_factory=list
    )  # Optional list of confidence scores for each bounding box
    masks: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Optional list of masks in RLE format

    image_metadata: ImageMetadata = field(default_factory=ImageMetadata)
