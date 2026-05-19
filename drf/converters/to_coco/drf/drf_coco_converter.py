import json
from typing import List
from pycocotools import mask as mask_utils

from drf.core.dtypes.prediction_2d import Prediction2D


class DRFCOCOConverter:
    """
    Convert DRF 2D prediction objects into a COCO-style JSON file.
    """

    def __init__(self):
        self.categories = {}
        self.category_list = []

    def _get_category_id(self, label: str) -> int:
        if label not in self.categories:
            new_id = len(self.categories) + 1
            self.categories[label] = new_id
            self.category_list.append(
                {"id": new_id, "name": label, "supercategory": ""}
            )
        return self.categories[label]

    def convert(self, predictions: List[Prediction2D], output_path: str):
        coco = {"images": [], "annotations": [], "categories": []}

        ann_id = 1
        for img_id, pred in enumerate(predictions, start=1):
            coco["images"].append(
                {
                    "id": img_id,
                    "file_name": pred.image_metadata.name,
                    "width": pred.image_metadata.width,
                    "height": pred.image_metadata.height,
                }
            )

            for i, bbox in enumerate(pred.bboxes):
                label = pred.labels[i]
                cat_id = self._get_category_id(label)

                ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": (
                        float(mask_utils.area(pred.masks[i]))
                        if pred.masks
                        else bbox[2] * bbox[3]
                    ),
                    "segmentation": pred.masks[i],
                    "score": pred.bbox_scores[i] if pred.bbox_scores else 1.0,
                    "iscrowd": 0,
                }
                coco["annotations"].append(ann)
                ann_id += 1

        coco["categories"] = self.category_list

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=4)
