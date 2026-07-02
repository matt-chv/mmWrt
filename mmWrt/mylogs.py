import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa F401
from functools import wraps

formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
default_formatter = logging.Formatter(formatter_str)  # noqa E501

default_logger = logging.getLogger("default")


def auto_log(func):
    # wrapper to easily add hiearichal logging down to class methods
    logger = logging.getLogger(f"{func.__module__}.{func.__qualname__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = logger  # local variable
        return func(*args, log=log, **kwargs)

    return wrapper
