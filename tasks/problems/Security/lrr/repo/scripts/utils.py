"""Utility functions"""

import errno
import functools
from itertools import islice
import logging
import time
from typing import Callable, Iterable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def batched(
    iterable: Iterable[T], n: int, *, strict: bool = False
) -> Iterable[Iterable[T]]:
    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def retry_on_blockingio(
    retries: int = 5, delay: float = 0.5, backoff_factor: float = 2.0
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception = None

            for i in range(retries):
                try:
                    return func(*args, **kwargs)

                except BlockingIOError as e:
                    if e.errno == errno.EAGAIN:
                        last_exception = e
                        logging.warning("%s: %s", func.__name__, e)
                        logging.warning(
                            "Retry %d, wait: %.2fs", i, current_delay
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        raise e
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator
