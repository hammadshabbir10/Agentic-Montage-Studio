import logging

_LOGGER_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _LOGGER_CONFIGURED
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)
