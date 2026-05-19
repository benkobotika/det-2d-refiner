import json
import argparse
from pathlib import Path


class CocoGtPredMatcher:
    """
    Align COCO prediction IDs and categories to a Ground Truth file.
    """

    def __init__(self, gt_path: str, pred_path: str):
        self.gt_data = self._load_json(gt_path)
        self.pred_data = self._load_json(pred_path)

        self.filename_to_gt_img_id = {
            img["file_name"]: img["id"] for img in self.gt_data["images"]
        }
        self.cat_name_to_gt_cat_id = {
            cat["name"]: cat["id"] for cat in self.gt_data["categories"]
        }
        self.gt_category_names = set(self.cat_name_to_gt_cat_id.keys())

    def _load_json(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def map_predictions(self) -> dict:
        new_annotations = []

        pred_cat_id_to_name = {
            cat["id"]: cat["name"] for cat in self.pred_data["categories"]
        }

        pred_img_id_to_filename = {
            img["id"]: img["file_name"] for img in self.pred_data["images"]
        }

        for idx, ann in enumerate(self.pred_data["annotations"]):
            cat_name = pred_cat_id_to_name[ann["category_id"]]
            if cat_name not in self.gt_category_names:
                continue

            new_ann = ann.copy()
            new_ann["id"] = idx + 1
            new_ann["image_id"] = self.filename_to_gt_img_id[
                pred_img_id_to_filename[ann["image_id"]]
            ]
            new_ann["category_id"] = self.cat_name_to_gt_cat_id[cat_name]
            new_annotations.append(new_ann)

        return {
            "images": self.gt_data["images"],
            "annotations": new_annotations,
            "categories": self.gt_data["categories"],
        }

    def save(self, output_path: str, data: dict):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Align COCO prediction IDs and categories to a Ground Truth file."
    )
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to the ground truth COCO JSON."
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to the predictions COCO JSON."
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to save the aligned COCO JSON.",
    )

    args = parser.parse_args()

    mapper = CocoGtPredMatcher(args.gt, args.pred)
    aligned_data = mapper.map_predictions()
    mapper.save(args.out, aligned_data)


if __name__ == "__main__":
    main()
