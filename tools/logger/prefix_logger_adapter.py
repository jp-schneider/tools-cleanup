import logging
from typing import Any, Dict, Tuple


class PrefixLoggerAdapter(logging.LoggerAdapter):
    """Logging adapter to add a prefix to the log message"""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]) -> None:
        """Creates a new PrefixLoggerAdapter instance.

        Will add a prefix, within extra to the log message.

        Example:
        ```python
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger = PrefixLoggerAdapter(logger, dict(prefix="MyPrefix"))
        logger.info("Hello World!")
        ```

        Parameters
        ----------
        logger : logging.Logger
            The logger to adapt.
        extra : Dict[str, Any]
            The extra dictionary containing the prefix.
        """
        return super().__init__(logger=logger, extra=extra)

    def process(self, msg: str, kwargs: dict) -> Tuple[str, dict]:
        if "prefix" not in self.extra or len(self.extra["prefix"]) == 0:
            return (msg, kwargs)
        else:
            return (f'[{self.extra["prefix"]}] ' + msg, kwargs)
