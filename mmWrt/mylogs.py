import logging
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from functools import wraps

def auto_log(func):
    logger = logging.getLogger(f"{func.__module__}.{func.__qualname__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        log = logger  # local variable
        return func(*args, log=log, **kwargs)

    return wrapper