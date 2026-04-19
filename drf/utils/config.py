from addict import Dict
from pathlib import Path
import yaml


class Config:
    """
    Configuration loader and accessor.
    This class loads a YAML configuration file and provides attribute-style access
    to its contents.
    """

    def __init__(self, config_file_path: str | Path):
        with open(config_file_path, "r") as file:
            data = yaml.safe_load(file)
        self._config = Dict(data)

    def __getattr__(self, key: str):
        return getattr(self._config, key)
