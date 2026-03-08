from dataclasses import dataclass

@dataclass
class ImageMetadata:
    name: str = ""
    width: int = 0
    height: int = 0