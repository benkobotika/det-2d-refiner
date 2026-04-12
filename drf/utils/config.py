from addict import Dict
from pathlib import Path
import yaml


class Config:
    def __init__(self, config_file_path: str | Path):
        with open(config_file_path, "r") as file:
            data = yaml.safe_load(file)
        self._config = Dict(data)

    def __getattr__(self, key: str):
        return getattr(self._config, key)
