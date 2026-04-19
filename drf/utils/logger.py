import logging


class Logger:
    """
    Simple wrapper around Python's built-in logging module to provide consistent formatting and easy instantiation.
    """

    def __init__(self, logger_name: str = __name__, level: int = logging.INFO):
        self._log = logging.getLogger(logger_name)
        self._log.setLevel(level)
        if not self._log.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(levelname)s] %(asctime)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._log.addHandler(handler)

    def debug(self, message, *args, **kwargs):
        self._log.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self._log.info(message, *args, **kwargs)

    def warn(self, message, *args, **kwargs):
        self._log.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._log.error(message, *args, **kwargs)
