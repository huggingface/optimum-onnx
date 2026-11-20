from pathlib import Path

from optimum.utils import logging


logger = logging.get_logger(__name__)


def log_earlier_validation_errors(exceptions: list[tuple[Path, BaseException]]) -> None:
    """Log every validation failure except the last one.

    The caller raises the last error. Indexing the `(path, error)` tuple with
    the enumerate index crashes once a third model fails, and it logs the path
    instead of the error for the first one.
    """
    for path, error in exceptions[:-1]:
        logger.error(f"Validation for the model {path.as_posix()} raised: {error}")
