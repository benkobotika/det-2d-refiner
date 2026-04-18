import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from tqdm import tqdm
from pycocotools import mask as mask_utils

from drf.core.img_service_runner import ImgServiceRunner
from drf.core.adapters.rfdetr import RFDETR
from drf.core.adapters.sam3 import SAM3
from drf.core.adapters.real_esrgan import RealESRGAN

from drf.converters.to_coco.drf.drf_coco_converter import DRFCOCOConverter
from drf.core.dtypes.prediction_2d import Prediction2D
from drf.core.dtypes.image_metadata import ImageMetadata
from drf.utils.config import Config
from drf.utils.logger import Logger


class Inference:
    def __init__(self, args: argparse.Namespace):
        # Store arguments
        self.args = args

        # Load configuration
        self.config = Config(Path(__file__).parent / "config.yaml")
        self.logger = Logger()

        # Initialize the image service runner
        self.runner = ImgServiceRunner()

        # Hyperparameters
        self.box_iou_threshold = args.box_iou_threshold
        self.mask_iou_threshold = args.mask_iou_threshold
        self.mask_voting_threshold = args.mask_voting_threshold
        self.use_tta = args.tta
        self.use_upscale = args.upscale

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

        # 1. Optional Upscaling for images smaller than 1000x1000
        if self.use_upscale and img.size[0] <= 1000 and img.size[1] <= 1000:
            model_scale, img = self.runner.run(self.config.upscaler, img)
        else:
            model_scale = 1.0

        # 2. Generate image variants (TTA)
        variants = self._get_image_variants(img)
        num_variants = len(variants)

        # 3. Run inference on all variants
        for current_img, is_flipped, scale in variants:
            for svc_name in self.config.detectors:
                pred = self.runner.run(
                    svc_name,
                    current_img,
                    **self.config.services[svc_name].run_kwargs,
                )

                if is_flipped or scale * model_scale != 1.0:
                    pred = self._normalize_coordinates(
                        pred, orig_w, orig_h, is_flipped, scale * model_scale
                    )

                self._append_predictions(final_pred, pred)

        # 4. Apply Mask Fusion
        return self._apply_mask_fusion(final_pred, num_variants)

    def _get_image_variants(self, img):
        variants = [(img, False, 1.0)]
        if not self.use_tta:
            return variants

        # Horizontal Flip
        variants.append((ImageOps.mirror(img), True, 1.0))

        # Multi-Scale
        for scale in [0.75, 1.0, 1.25, 1.5]:
            new_size = (int(img.width * scale), int(img.height * scale))
            variants.append(
                (
                    img.resize(new_size, Image.Resampling.BILINEAR),
                    False,
                    scale,
                )
            )

        # Brightness
        enhancer = ImageEnhance.Brightness(img)
        variants.extend(
            [
                (enhancer.enhance(0.7), False, 1.0),
                (enhancer.enhance(1.3), False, 1.0),
            ]
        )

        # Contrast
        contrast = ImageEnhance.Contrast(img)
        variants.extend(
            [
                (contrast.enhance(0.8), False, 1.0),
                (contrast.enhance(1.2), False, 1.0),
            ]
        )

        # Color Saturation
        enhancer_color = ImageEnhance.Color(img)
        variants.append((enhancer_color.enhance(1.3), False, 1.0))

        # Blur
        variants.append((img.filter(ImageFilter.GaussianBlur(1)), False, 1.0))

        # Sharpening
        variants.append(
            (img.filter(ImageFilter.UnsharpMask(radius=2, percent=150)), False, 1.0)
        )

        return variants

    def _normalize_coordinates(
        self,
        pred: Prediction2D,
        orig_w: int,
        orig_h: int,
        is_flipped: bool,
        scale: float,
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
            if m.shape != (orig_h, orig_w):
                m = Image.fromarray(m).resize(
                    (orig_w, orig_h),
                    Image.Resampling.BILINEAR,
                )
                m = (np.array(m) > 0.5).astype(np.uint8)

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

    def _compute_box_iou(self, box1: list, box2: list) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1 + w1, x2 + w2)
        yy2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, xx2 - xx1)
        inter_h = max(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / (union_area + 1e-6)

    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _apply_mask_fusion(
        self, predictions: Prediction2D, num_variants: int
    ) -> Prediction2D:
        if not predictions.bboxes:
            return predictions

        unique_labels = set(predictions.labels)
        all_decoded_masks = [mask_utils.decode(m) for m in predictions.masks]

        fused_pred = Prediction2D(image_metadata=predictions.image_metadata)
        for label in unique_labels:
            indices = [i for i, l in enumerate(predictions.labels) if l == label]
            if not indices:
                continue

            boxes = np.array([predictions.bboxes[i] for i in indices])
            scores = np.array([predictions.bbox_scores[i] for i in indices])
            decoded_masks = [all_decoded_masks[i] for i in indices]

            sort_idx = np.argsort(scores)[::-1]
            boxes = boxes[sort_idx]
            scores = scores[sort_idx]
            decoded_masks = [decoded_masks[i] for i in sort_idx]

            groups = []
            for i in range(len(boxes)):
                matched = False
                for g in groups:
                    if (
                        self._compute_box_iou(boxes[i], boxes[g[0]])
                        > self.box_iou_threshold
                    ):
                        if (
                            self._compute_mask_iou(
                                decoded_masks[i], decoded_masks[g[0]]
                            )
                            > self.mask_iou_threshold
                        ):
                            g.append(i)
                            matched = True
                            break
                if not matched:
                    groups.append([i])

            for g in groups:
                g_scores = scores[g]
                g_masks_decoded = np.array([decoded_masks[i] for i in g])
                weights = g_scores.reshape(-1, 1, 1)
                sum_weights = np.sum(weights)
                weighted_sum_mask = np.sum(g_masks_decoded * weights, axis=0)
                avg_mask = weighted_sum_mask / sum_weights

                fused_mask_np = (avg_mask >= self.mask_voting_threshold).astype(
                    np.uint8
                )

                if not np.any(fused_mask_np):
                    continue

                fused_score = np.mean(g_scores)

                fused_mask = mask_utils.encode(np.asfortranarray(fused_mask_np))
                fused_mask["counts"] = fused_mask["counts"].decode("utf-8")

                fused_box = mask_utils.toBbox(fused_mask).tolist()

                fused_pred.bboxes.append(fused_box)
                fused_pred.bbox_scores.append(float(fused_score))
                fused_pred.labels.append(label)
                fused_pred.masks.append(fused_mask)

        return fused_pred

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
        description="Inference-time object detection quality optimization pipeline"
    )
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="Input images directory"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True, help="Output COCO JSON path"
    )
    parser.add_argument(
        "--box_iou_threshold",
        type=float,
        default=0.50,
        help="Box IoU threshold for preliminary grouping",
    )
    parser.add_argument(
        "--mask_iou_threshold",
        type=float,
        default=0.75,
        help="Mask IoU threshold for precise pixel grouping",
    )
    parser.add_argument(
        "--mask_voting_threshold",
        type=float,
        default=0.4,
        help="Pixel voting threshold for mask fusion",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable TTA (Test-Time Augmentation)",
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Enable upscaling for images smaller than 1000x1000 (both dimensions)",
    )
    args = parser.parse_args()
    Inference(args).run()


if __name__ == "__main__":
    main()
