from typing import Type, Dict, Any
from PIL import Image


class ImgServiceRunner:
    def __init__(self):
        self._active_services: Dict[str, Any] = {}
        self._service_configs: Dict[str, Dict[str, Any]] = {}
        self._is_active: bool = False

    def add_service(self, service_name: str, adapter_class: Type, **kwargs):
        if service_name in self._active_services:
            old_instance = self._active_services[service_name]
            if hasattr(old_instance, "__exit__"):
                old_instance.__exit__(None, None, None)

        self._service_configs[service_name] = {"class": adapter_class, "kwargs": kwargs}

        if self._is_active:
            instance = adapter_class(**kwargs)
            if hasattr(instance, "__enter__"):
                instance.__enter__()
            self._active_services[service_name] = instance

    def remove_service(self, service_name: str):
        if service_name in self._active_services:
            instance = self._active_services[service_name]
            if hasattr(instance, "__exit__"):
                instance.__exit__(None, None, None)
            del self._active_services[service_name]

        if service_name in self._service_configs:
            del self._service_configs[service_name]

    def __enter__(self):
        self._is_active = True
        for name, config in self._service_configs.items():
            if name not in self._active_services:
                instance = config["class"](**config["kwargs"])
                if hasattr(instance, "__enter__"):
                    instance.__enter__()
                self._active_services[name] = instance
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_active = False
        for _, instance in self._active_services.items():
            if hasattr(instance, "__exit__"):
                instance.__exit__(exc_type, exc_val, exc_tb)
        self._active_services.clear()

    def run(self, service_name: str, img: Image.Image, **service_kwargs) -> Any:
        if service_name not in self._active_services:
            raise ValueError(f"Service '{service_name}' is not registered or active.")

        return self._active_services[service_name].run(img, **service_kwargs)
