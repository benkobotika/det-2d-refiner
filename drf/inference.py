import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
from pycocotools import mask as mask_utils

from drf.core.img_service_runner import ImgServiceRunner
from drf.core.adapters.rfdetr import RFDETR
from drf.core.adapters.sam3 import SAM3
from drf.core.adapters.real_esrgan import RealESRGAN

from drf.converters.to_coco.a2d2.drf_coco_converter import DRFCOCOConverter
from drf.core.dtypes.prediction_2d import Prediction2D
from drf.core.dtypes.image_metadata import ImageMetadata
from drf.utils.config import Config
from drf.utils.logger import Logger


class Inference:
    def __init__(self, args: argparse.Namespace):
        # Store arguments and load configuration
        self.args = args
        self.config = Config(args.config)
        self.logger = Logger()

        # Initialize the image service runner
        self.runner = ImgServiceRunner()

        self.iou_threshold = args.iou_threshold
        self.use_tta = args.tta

    def _setup_services(self):
        for svc_name, svc_data in self.config.services.items():
            adapter_class = globals().get(svc_data.adapter_class)
            if not adapter_class:
                self.logger.error(
                    f"Adapter class {svc_data.adapter_class} not found in globals."
                )
                continue

            self.runner.add_service(svc_name, adapter_class, **svc_data.init_kwargs)
            self.logger.info(f"Registered service: {svc_name}")

    def process_image(self, img: Image.Image) -> Prediction2D:
        orig_w, orig_h = img.size
        final_pred = Prediction2D(
            image_metadata=ImageMetadata(width=orig_w, height=orig_h)
        )

        variants = [(img, False, 1.0)]

        if self.use_tta:
            # 1. Horizontal Flip
            variants.append((ImageOps.mirror(img), True, 1.0))

            # 2. Multi-Scale
            for scale in [0.8, 1.2]:
                new_size = (int(orig_w * scale), int(orig_h * scale))
                scaled_img = img.resize(new_size, Image.Resampling.BILINEAR)
                variants.append((scaled_img, False, scale))

            # 3. Brightness adjustment
            enhancer = ImageEnhance.Brightness(img)
            variants.append((enhancer.enhance(0.8), False, 1.0))
            variants.append((enhancer.enhance(1.2), False, 1.0))

        for current_img, is_flipped, scale in variants:
            pred = self.runner.run(
                "rfdetr", current_img, **self.config.services.rfdetr.run_kwargs
            )

            if is_flipped or scale != 1.0:
                pred = self._normalize_coordinates(pred, orig_w, is_flipped, scale)

            self._append_predictions(final_pred, pred)

        return self._apply_nms(final_pred)

    def _normalize_coordinates(
        self, pred: Prediction2D, orig_w: int, is_flipped: bool, scale: float
    ) -> Prediction2D:
        new_bboxes = []
        for x, y, w, h in pred.bboxes:
            # Undo Scale
            nx, ny, nw, nh = x / scale, y / scale, w / scale, h / scale

            # Undo Flip
            if is_flipped:
                nx = orig_w - nx - nw
            new_bboxes.append([nx, ny, nw, nh])

        pred.bboxes = new_bboxes

        new_masks = []
        for mask in pred.masks:
            # Decode mask
            m = mask_utils.decode(mask)

            # Undo Scale
            if scale != 1.0:
                m_img = Image.fromarray(m).resize(
                    (orig_w, int(m.shape[0] / scale)), Image.Resampling.NEAREST
                )
                m = np.array(m_img)

            # Undo Flip
            if is_flipped:
                m = np.fliplr(m)

            # Re-encode mask
            m = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            m["counts"] = m["counts"].decode("utf-8")

            new_masks.append(m)

        pred.masks = new_masks

        return pred

    def _append_predictions(self, target: Prediction2D, source: Prediction2D):
        target.bboxes.extend(source.bboxes)
        target.labels.extend(source.labels)
        target.bbox_scores.extend(source.bbox_scores)
        target.masks.extend(source.masks)

    def _apply_nms(self, predictions: Prediction2D) -> Prediction2D:
        if not predictions.bboxes:
            return predictions

        boxes = np.array(predictions.bboxes)
        scores = np.array(predictions.bbox_scores)
        labels = np.array(predictions.labels)

        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        indices = np.argsort(scores)[::-1]
        keep_indices = []

        while indices.size > 0:
            current = indices[0]
            keep_indices.append(current)

            if indices.size == 1:
                break

            rest_indices = indices[1:]

            xx1 = np.maximum(x1[current], x1[rest_indices])
            yy1 = np.maximum(y1[current], y1[rest_indices])
            xx2 = np.minimum(x2[current], x2[rest_indices])
            yy2 = np.minimum(y2[current], y2[rest_indices])

            w, h = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[current] + areas[rest_indices] - inter + 1e-6)
            indices = rest_indices[iou <= self.iou_threshold]

        return Prediction2D(
            bboxes=boxes[keep_indices].tolist(),
            labels=labels[keep_indices].tolist(),
            bbox_scores=scores[keep_indices].tolist(),
            masks=[predictions.masks[i] for i in keep_indices],
            image_metadata=predictions.image_metadata,
        )

    def run(self):
        # Validate input directory
        input_dir = Path(self.args.input_dir)
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        image_paths = [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        if not image_paths:
            self.logger.error(f"No images found in {input_dir}")
            return

        # Setup services and run inference
        self._setup_services()
        predictions = []
        with self.runner:
            for img_path in tqdm(image_paths, desc="Processing images"):
                img = Image.open(img_path).convert("RGB")

                pred = self.process_image(img)
                pred.image_metadata.name = img_path.name
                predictions.append(pred)

        # Save results in COCO format
        output_file = Path(self.args.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        DRFCOCOConverter().convert(predictions, str(output_file))


def main():
    parser = argparse.ArgumentParser(
        description="Inference-time ML-model quality optimization pipeline."
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="Input images directory."
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Output COCO JSON path."
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config."
    )

    parser.add_argument(
        "--iou_threshold", type=float, default=0.75, help="IoU threshold for NMS."
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable Multi-stage TTA (Flip, Scale, Brightness)",
    )
    args = parser.parse_args()
    Inference(args).run()


if __name__ == "__main__":
    main()
