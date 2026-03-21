import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from drf.core.img_service_runner import ImgServiceRunner
from drf.core.adapters.sam3 import SAM3
from drf.core.adapters.rfdetr import RFDETR
from drf.converters.to_coco.a2d2.drf_coco_converter import DRFCOCOConverter
from drf.utils.config import Config
from drf.utils.logger import Logger


def main():
    # 1. Setup Argparser
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path where the output COCO JSON file will be saved.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    # 2. Load YAML Configuration
    config = Config(args.config)

    # 3. Initialize Logger
    logger = Logger("main")

    # 4. Read input images
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg"}
    image_paths = [
        p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions
    ]

    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images")

    # 5. Run Inference Pipeline
    predictions = []
    runner = ImgServiceRunner()

    with runner:
        for svc_name, svc_data in config.services.items():
            adapter_class = globals()[svc_data.adapter_class]
            init_kwargs = svc_data.init_kwargs

            runner.add_service(svc_name, adapter_class, **init_kwargs)
            logger.info(f"Registered service: {svc_name} on port {init_kwargs.port}")

        for img_path in tqdm(image_paths, desc="Processing images"):
            img = Image.open(img_path)
            pred = runner.run("rfdetr", img, **config.services.rfdetr.run_kwargs)
            pred.image_metadata.name = img_path.name
            predictions.append(pred)

    # 6. Save to COCO format
    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    converter = DRFCOCOConverter()
    converter.convert(predictions, str(output_file))


if __name__ == "__main__":
    main()
