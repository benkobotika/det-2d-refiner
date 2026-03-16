import sys
import io
import json
import argparse
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCOEvaluator:
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

    def __init__(self, gt_path, pred_path):
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

    def _extract_general_metrics(self, coco_eval):
        iou_idx = 5  # 0.75 IoU threshold index
        gt_count = tp = fp = 0
        all_area = coco_eval.params.areaRng[0]

        for ev in coco_eval.evalImgs:
            if ev is None:
                continue
            if ev["aRng"] != all_area:
                continue

            gt_count += len(ev["gtIds"])
            dt_matches = ev["dtMatches"][iou_idx]
            for match in dt_matches:
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

    def run(self, iou_types=["bbox", "segm"]):
        for iou_type in iou_types:
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Global Evaluation
                coco_eval = COCOeval(self.coco_gt, self.coco_pred, iou_type)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                self.class_results.append(self._extract_stats(coco_eval, iou_type))

                # Per-Class Evaluation
                cat_ids = self.coco_gt.getCatIds()
                categories = {
                    cat["id"]: cat["name"] for cat in self.coco_gt.loadCats(cat_ids)
                }

                for c_id, c_name in categories.items():
                    c_eval = COCOeval(self.coco_gt, self.coco_pred, iou_type)
                    c_eval.params.catIds = [c_id]
                    c_eval.evaluate()
                    c_eval.accumulate()
                    c_eval.summarize()
                    self.class_results.append(
                        self._extract_stats(c_eval, iou_type, c_name)
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

    args = parser.parse_args()

    evaluator = COCOEvaluator(args.gt, args.pred)
    evaluator.run()
    evaluator.save_results(args.out)


if __name__ == "__main__":
    main()
