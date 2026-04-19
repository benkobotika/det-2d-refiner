import sys
import io
import os
import json
import argparse
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
    """
    Evaluate COCO-style detection/segmentation predictions globally, per class, and per image.
    This evaluator wraps pycocotools.COCOeval and augments standard COCO summary
    statistics with additional count-based metrics computed at a configurable IoU threshold
    (default: 0.75), including TP/FP/FN, precision, and recall.
    """

    COCO_METRIC_NAMES = [
        "AP_50_95",
        "AP_50",
        "AP_75",
        "AP_S",
        "AP_M",
        "AP_L",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_S",
        "AR_M",
        "AR_L",
    ]

    def __init__(self, gt_path, pred_path, size_thresholds=[32, 96]):
        self.size_thresholds = size_thresholds
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self.coco_gt = COCO(gt_path)
            with open(pred_path, "r") as f:
                preds = json.load(f)["annotations"]
            self.coco_pred = self.coco_gt.loadRes(preds)
        finally:
            sys.stdout = original_stdout

        self.class_results = []
        self.image_results = []

    def _set_custom_bins(self, coco_eval):
        s, m = self.size_thresholds
        coco_eval.params.areaRng = [
            [0**2, 1e5**2],  # All
            [0**2, s**2],  # Small
            [s**2, m**2],  # Medium
            [m**2, 1e5**2],  # Large
        ]
        coco_eval.params.areaRngLbl = ["all", "small", "medium", "large"]

    def _extract_general_metrics(self, coco_eval, iou_threshold=0.75):
        iou_idx = min(
            range(len(coco_eval.params.iouThrs)),
            key=lambda i: abs(coco_eval.params.iouThrs[i] - iou_threshold),
        )
        gt_count = tp = fp = 0
        all_area = coco_eval.params.areaRng[0]

        for ev in coco_eval.evalImgs:
            if ev is None or ev["aRng"] != all_area:
                continue

            gt_ignore = ev["gtIgnore"]
            gt_count += sum(1 for ignored in gt_ignore if not ignored)

            dt_matches = ev["dtMatches"][iou_idx]
            dt_ignore = ev["dtIgnore"][iou_idx]
            for match, ignored in zip(dt_matches, dt_ignore):
                if ignored:
                    continue
                if match > 0:
                    tp += 1
                else:
                    fp += 1

        fn = gt_count - tp
        pred_count = tp + fp
        precision = tp / pred_count if pred_count > 0 else 1
        recall = tp / gt_count if gt_count > 0 else 1

        return {
            "GT": gt_count,
            "PRED": pred_count,
            "TP_75": tp,
            "FP_75": fp,
            "FN_75": fn,
            "Precision_75": precision,
            "Recall_75": recall,
        }

    def _extract_stats(self, coco_eval, iou_type, cls="all"):
        stats = {"IoU_Type": iou_type, "Class": cls}
        stats.update(self._extract_general_metrics(coco_eval))
        stats.update(
            {name: coco_eval.stats[i] for i, name in enumerate(self.COCO_METRIC_NAMES)}
        )
        return stats

    def _extract_per_image_general_metrics(
        self, coco_eval, iou_type, cls="all", iou_threshold=0.75
    ):
        iou_idx = min(
            range(len(coco_eval.params.iouThrs)),
            key=lambda i: abs(coco_eval.params.iouThrs[i] - iou_threshold),
        )
        all_area = coco_eval.params.areaRng[0]
        per_image = {}

        for ev in coco_eval.evalImgs:
            if ev is None or ev["aRng"] != all_area:
                continue

            img_id = ev["image_id"]
            if img_id not in per_image:
                img_info = self.coco_gt.imgs.get(img_id, {})
                per_image[img_id] = {
                    "IoU_Type": iou_type,
                    "Class": cls,
                    "File_Name": img_info.get("file_name", ""),
                    "GT": 0,
                    "PRED": 0,
                    "TP_75": 0,
                    "FP_75": 0,
                }

            img_metrics = per_image[img_id]

            gt_ignore = ev["gtIgnore"]
            img_metrics["GT"] += sum(1 for ignored in gt_ignore if not ignored)

            dt_matches = ev["dtMatches"][iou_idx]
            dt_ignore = ev["dtIgnore"][iou_idx]
            for match, ignored in zip(dt_matches, dt_ignore):
                if ignored:
                    continue
                img_metrics["PRED"] += 1
                if match > 0:
                    img_metrics["TP_75"] += 1
                else:
                    img_metrics["FP_75"] += 1

        rows = []
        for img_metrics in per_image.values():
            img_metrics["FN_75"] = img_metrics["GT"] - img_metrics["TP_75"]
            img_metrics["Precision_75"] = (
                img_metrics["TP_75"] / img_metrics["PRED"]
                if img_metrics["PRED"] > 0
                else 1
            )
            img_metrics["Recall_75"] = (
                img_metrics["TP_75"] / img_metrics["GT"] if img_metrics["GT"] > 0 else 1
            )
            rows.append(img_metrics)

        return rows

    def run(self, iou_types=["bbox", "segm"]):
        for iou_type in iou_types:
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Global Evaluation
                coco_eval = COCOeval(self.coco_gt, self.coco_pred, iou_type)
                self._set_custom_bins(coco_eval)

                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                self.class_results.append(self._extract_stats(coco_eval, iou_type))
                self.image_results.extend(
                    self._extract_per_image_general_metrics(coco_eval, iou_type)
                )

                # Per-Class Evaluation
                cat_ids = self.coco_gt.getCatIds()
                categories = {
                    cat["id"]: cat["name"] for cat in self.coco_gt.loadCats(cat_ids)
                }

                for c_id, c_name in categories.items():
                    c_eval = COCOeval(self.coco_gt, self.coco_pred, iou_type)
                    self._set_custom_bins(c_eval)

                    c_eval.params.catIds = [c_id]
                    c_eval.evaluate()
                    c_eval.accumulate()
                    c_eval.summarize()
                    self.class_results.append(
                        self._extract_stats(c_eval, iou_type, c_name)
                    )
                    self.image_results.extend(
                        self._extract_per_image_general_metrics(
                            c_eval, iou_type, cls=c_name
                        )
                    )
            finally:
                sys.stdout = original_stdout

    def save_results(self, output_csv):
        if self.class_results:
            df = pd.DataFrame(self.class_results)
            cols = [
                "IoU_Type",
                "Class",
                "GT",
                "PRED",
                "TP_75",
                "FP_75",
                "FN_75",
                "Precision_75",
                "Recall_75",
            ] + self.COCO_METRIC_NAMES
            df = df[cols]
            df.to_csv(output_csv, index=False)

        if self.image_results:
            df = pd.DataFrame(self.image_results)
            cols = [
                "IoU_Type",
                "Class",
                "File_Name",
                "GT",
                "PRED",
                "TP_75",
                "FP_75",
                "FN_75",
                "Precision_75",
                "Recall_75",
            ]
            df = df[cols]
            img_output_csv = os.path.splitext(output_csv)[0] + "_per_image.csv"
            df.to_csv(img_output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="COCO Evaluation Script")
    parser.add_argument(
        "--gt", type=str, required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--pred", type=str, required=True, help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--out", type=str, default="evaluation_results.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--size_thresholds",
        type=int,
        nargs=2,
        default=[32, 96],
        help="Two integers defining the square root of area for S/M and M/L boundaries."
        "Default: 32 96 (standard COCO).",
    )

    args = parser.parse_args()

    evaluator = COCOEvaluator(args.gt, args.pred, size_thresholds=args.size_thresholds)
    evaluator.run()
    evaluator.save_results(args.out)


if __name__ == "__main__":
    main()
