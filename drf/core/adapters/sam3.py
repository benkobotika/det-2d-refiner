import httpx
import json
import io
import PIL.Image
from typing import Dict, Any

from drf.core.dtypes.prediction_2d import Prediction2D
from drf.core.dtypes.image_metadata import ImageMetadata

class SAM3:
    def __init__(self, base_url: str = None, port: int = 8002):
        if base_url is None:
            base_url = f"http://localhost:{port}"
        self.inference_url = f"{base_url.rstrip('/')}/segment"
        self.client = httpx.Client(timeout=60.0)

    def inference(
        self, 
        image: PIL.Image.Image,
        prompt: str = None, 
        box: list = None
    ) -> Prediction2D:
        data = {}
        if prompt:
            data["text_prompt"] = prompt
            label = prompt
        elif box:
            data["box_prompt"] = json.dumps(box)
            label = None
        else:
            raise ValueError("Provide 'prompt' or 'box'")

        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format='JPEG')
        img_bytes_io.seek(0)

        files = {"image": ("img.jpg", img_bytes_io, "image/jpeg")}
        
        response = self.client.post(self.inference_url, files=files, data=data)
        response.raise_for_status()
            
        return self._adapt(response.json(), image, label)

    def _adapt(self, data: Dict[str, Any], image: PIL.Image.Image, label: str) -> Prediction2D:
        bboxes = data.get("boxes", [])
        return Prediction2D(
            bboxes=bboxes,
            labels=[label] * len(bboxes),
            bbox_scores=data.get("scores", []),
            masks=data.get("masks", []),
            image_metadata=ImageMetadata(width=image.width, height=image.height)
        )

    def close(self):
        self.client.close()