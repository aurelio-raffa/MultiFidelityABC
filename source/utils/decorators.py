import time

from functools import wraps


def time_it(only_time=False):
    def inner_function(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            t = time.time()
            res = function(*args, **kwargs)
            t = time.time() - t
            if only_time:
                return t
            else:
                return res, t
        return wrapper
    return inner_function


class CountIt:
    def __init__(self, function):
        self._funct = function
        self._calls = 0

    def __call__(self, *args, **kwargs):
        self._calls += 1
        return self._funct(*args, **kwargs)

    def get_calls(self):
        return self._calls

    def reset_calls(self):
        self._calls = 0

