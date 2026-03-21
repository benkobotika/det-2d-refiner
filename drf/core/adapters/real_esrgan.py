import httpx
import io
import PIL.Image


class RealESRGAN:
    def __init__(self, base_url: str = None, port: int = 8001):
        if base_url is None:
            base_url = f"http://127.0.0.1:{port}"

        self.enhance_url = f"{base_url.rstrip('/')}/enhance"
        self.client = httpx.Client(timeout=300.0)

    def run(self, image: PIL.Image.Image) -> PIL.Image.Image:
        img_bytes_io = io.BytesIO()
        image.save(img_bytes_io, format="PNG")
        img_bytes_io.seek(0)

        files = {"image": ("input.png", img_bytes_io, "image/png")}

        response = self.client.post(self.enhance_url, files=files)
        response.raise_for_status()

        upscaled_image = PIL.Image.open(io.BytesIO(response.content))
        return upscaled_image

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
