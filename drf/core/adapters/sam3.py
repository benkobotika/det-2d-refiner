import httpx
import io
import PIL.Image
from typing import Dict, Any

from drf.core.dtypes.prediction_2d import Prediction2D
from drf.core.dtypes.image_metadata import ImageMetadata


class SAM3:
    def __init__(self, base_url: str = None, port: int = 8002):
        if base_url is None:
            base_url = f"http://127.0.0.1:{port}"
        self.inference_url = f"{base_url.rstrip('/')}/segment"
        self.client = httpx.Client(timeout=60.0)

    def run(
        self,
        image: PIL.Image.Image,
        prompt: str,
    ) -> Prediction2D:
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format="JPEG")
        img_bytes_io.seek(0)

        files = {"image": ("img.jpg", img_bytes_io, "image/jpeg")}
        params = {"text_prompt": prompt}

        response = self.client.post(self.inference_url, files=files, params=params)
        response.raise_for_status()

        return self._adapt(response.json(), image)

    def _adapt(self, data: Dict[str, Any], image: PIL.Image.Image) -> Prediction2D:
        return Prediction2D(
            bboxes=data.get("boxes", []),
            labels=data.get("classes", []),
            bbox_scores=data.get("scores", []),
            masks=data.get("masks", []),
            image_metadata=ImageMetadata(width=image.width, height=image.height),
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
