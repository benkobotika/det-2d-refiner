import os
import argparse
import numpy as np
import cv2
import shutil
from tqdm import tqdm
from pycocotools.coco import COCO

from drf.utils.logger import Logger


class CocoVisualizer:
    """
    Visualize and compare COCO-format annotations across one or more annotation files.
    This visualizer loads images from image_dir, renders segmentation masks and/or
    bounding boxes with category labels, and writes stacked comparison panels to
    output_dir. Each output image contains one panel per COCO file, with a header
    showing the source annotation file name.
    """

    logger = Logger("CocoVisualizer")

    def __init__(
        self,
        image_dir: str,
        output_dir: str,
        classes: list[str] | None = None,
        alpha: float = 0.4,
        show_masks: bool = True,
        show_boxes: bool = True,
        instance_mode: bool = False,
    ):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.classes = classes
        self.alpha = alpha
        self.show_masks = show_masks
        self.show_boxes = show_boxes
        self.instance_mode = instance_mode
        self.category_colors = {}

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _get_color(self, seed_id: int) -> tuple:
        hue = (seed_id * 137.5) % 180  # using golden angle to distribute colors
        hsv = np.uint8([[[hue, 200, 255]]])  # high saturation and value
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr))

    def _get_cat_color(self, cat_id: int) -> tuple:
        if cat_id not in self.category_colors:
            self.category_colors[cat_id] = self._get_color(cat_id)
        return self.category_colors[cat_id]

    def _get_text_color(self, bg_color: tuple) -> tuple:
        brightness = (
            bg_color[0] * 0.10 + bg_color[1] * 0.60 + bg_color[2] * 0.30
        )  # calculate perceived brightness
        return (0, 0, 0) if brightness > 128 else (255, 255, 255)

    def _draw_annotations(
        self, image: np.ndarray, annotations: list, coco: COCO, title: str
    ) -> np.ndarray:
        vis_image = image.copy()

        # 1. Draw Masks
        if self.show_masks:
            mask_overlay = vis_image.copy()
            for ann in annotations:
                if "segmentation" in ann:
                    mask = coco.annToMask(ann)
                    color_id = ann["id"] if self.instance_mode else ann["category_id"]
                    color = self._get_color(color_id)
                    mask_overlay[mask == 1] = color
            cv2.addWeighted(
                mask_overlay, self.alpha, vis_image, 1 - self.alpha, 0, vis_image
            )

        # 2. Draw Boxes and Labels
        if self.show_boxes:
            for ann in annotations:
                if "bbox" in ann:
                    cat_id = ann["category_id"]
                    color = self._get_cat_color(cat_id)
                    text_color = self._get_text_color(color)

                    cat_info = coco.loadCats([cat_id])[0]
                    label = cat_info.get("name", f"ID:{cat_id}").upper()

                    x, y, w, h = map(int, ann["bbox"])
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale, thickness = 0.45, 1
                    (t_w, t_h), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    label_y = max(y, t_h + baseline + 5)
                    cv2.rectangle(
                        vis_image,
                        (x, label_y - t_h - baseline - 5),
                        (x + t_w + 2, label_y),
                        color,
                        -1,
                    )
                    cv2.putText(
                        vis_image,
                        label,
                        (x + 1, label_y - baseline - 2),
                        font,
                        font_scale,
                        text_color,
                        thickness,
                        cv2.LINE_AA,
                    )

        # 3. Add Header
        header_h = 40
        header = np.full(
            (header_h, vis_image.shape[1], 3), (50, 50, 50), dtype=np.uint8
        )
        cv2.putText(
            header,
            title,
            (10, 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return cv2.vconcat([header, vis_image])

    def run(self, coco_paths: list[str]):
        coco_objects = {os.path.basename(p): COCO(p) for p in coco_paths}
        ref_coco = list(coco_objects.values())[0]
        img_ids = ref_coco.getImgIds()

        for img_id in tqdm(img_ids, desc="Processing images"):
            img_info = ref_coco.loadImgs(img_id)[0]
            file_name = img_info["file_name"]
            img_path = os.path.join(self.image_dir, file_name)

            original_img = cv2.imread(img_path)

            panels = []
            for name, coco in coco_objects.items():
                ann_ids = coco.getAnnIds(imgIds=img_id)
                annotations = coco.loadAnns(ann_ids)
                if self.classes is not None:
                    annotations = [
                        ann
                        for ann in annotations
                        if coco.cats[ann["category_id"]]["name"] in self.classes
                    ]
                panel = self._draw_annotations(original_img, annotations, coco, name)
                panels.append(panel)

            stacked_vis = cv2.vconcat(panels)
            save_path = os.path.join(self.output_dir, file_name)
            cv2.imwrite(save_path, stacked_vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coco_files",
        nargs="+",
        required=True,
        help="List of COCO annotation files to visualize",
    )
    parser.add_argument(
        "--image_dir", required=True, help="Directory containing the images"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the visualized images"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, help="Transparency value for masks"
    )
    parser.add_argument(
        "--no_masks", action="store_true", help="Disable mask visualization"
    )
    parser.add_argument(
        "--no_boxes", action="store_true", help="Disable bounding box visualization"
    )
    parser.add_argument(
        "--instance_mode",
        action="store_true",
        help="Color masks by instance ID instead of category",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Optional list of category names to visualize (if not provided, all categories will be shown)",
    )
    args = parser.parse_args()

    visualizer = CocoVisualizer(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        alpha=args.alpha,
        show_masks=not args.no_masks,
        show_boxes=not args.no_boxes,
        instance_mode=args.instance_mode,
    )
    visualizer.run(args.coco_files)


if __name__ == "__main__":
    main()
