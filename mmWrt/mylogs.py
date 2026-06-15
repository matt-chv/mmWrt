import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from functools import wraps

default_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def auto_log(func):
    # wrapper to easily add hiearichal logging down to class methods
    logger = logging.getLogger(f"{func.__module__}.{func.__qualname__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = logger  # local variable
        return func(*args, log=log, **kwargs)

    return wrapper
