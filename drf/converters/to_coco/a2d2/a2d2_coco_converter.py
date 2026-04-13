import os
import json
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

from drf.utils.logger import Logger


class A2D2CocoConverter:
    logger = Logger("A2D2CocoConverter")

    # Mapping of class IDs to category names based on the documentation
    CLASS_IDX_MAP = {
        1: "car",
        2: "pedestrian",
        3: "truck",
        4: "smallVehicle",
        5: "utilityVehicle",
        6: "bicycle",
        7: "tractor",
    }

    def __init__(self, root_dir: str, target_folder_segment: str):
        self.root_dir = root_dir
        self.target_folder_segment = target_folder_segment
        self.categories = self._create_category_annotations()

    def _create_category_annotations(self):
        categories = []
        for idx, name in self.CLASS_IDX_MAP.items():
            categories.append({"id": idx, "name": name, "supercategory": ""})
        return categories

    def _process_folder(self, folder_path: str) -> str:
        coco_output = {"images": [], "annotations": [], "categories": self.categories}
        image_id_counter = 1
        annotation_id_counter = 1

        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])

        for image_filename in tqdm(
            image_files, desc=f"Processing images in {folder_path}"
        ):
            image_path = os.path.join(folder_path, image_filename)

            image = Image.open(image_path)
            image_np = np.array(image, dtype=np.uint16)
            height, width = image_np.shape

            coco_output["images"].append(
                {
                    "id": image_id_counter,
                    "width": width,
                    "height": height,
                    "file_name": image_filename.replace("_instance_", "_camera_"),
                }
            )

            unique_values = np.unique(image_np)

            for val in unique_values:
                if val == 0:
                    continue  # skip background

                class_id = int(
                    val >> 10
                )  # extract class ID from the 16-bit value (class ID is stored in the upper 6 bits)

                if class_id not in self.CLASS_IDX_MAP:
                    raise ValueError(
                        f"Unknown class ID {class_id} found in image {image_filename}"
                    )

                mask = (image_np == val).astype(np.uint8)

                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")

                annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": class_id,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": mask_utils.toBbox(rle).tolist(),
                    "iscrowd": 0,
                }

                coco_output["annotations"].append(annotation)
                annotation_id_counter += 1

            image_id_counter += 1

        output_path = os.path.join(folder_path, "coco_annotations.json")
        with open(output_path, "w") as f:
            json.dump(coco_output, f, indent=4)

        return output_path

    def convert(self):
        found_folders = []
        for root, _, _ in os.walk(self.root_dir):
            if self.target_folder_segment in root and root.endswith(
                self.target_folder_segment
            ):
                found_folders.append(root)

        if not found_folders:
            self.logger.warn(
                f"No folders containing '{self.target_folder_segment}' found in '{self.root_dir}'."
            )
            return

        for folder in found_folders:
            self._process_folder(folder)


def main():
    parser = argparse.ArgumentParser(
        description="A2D2 16-bit Instance to COCO Converter"
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--target_segment",
        type=str,
        default=os.path.join("instance", "cam_front_center"),
    )
    args = parser.parse_args()

    converter = A2D2CocoConverter(
        root_dir=args.root_dir, target_folder_segment=args.target_segment
    )
    converter.convert()


if __name__ == "__main__":
    main()
