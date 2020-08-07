import datetime
import time
from functools import wraps


def timing(f):
    '''
    Wrapper function to time function execution
    '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'    {f.__name__}: {datetime.timedelta(seconds=(te - ts))}')
        return result
    return wrap
