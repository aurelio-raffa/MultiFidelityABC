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
