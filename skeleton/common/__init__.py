from os import getenv

def _strtobool (val):
    """
    Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    This is taken from distutils, which is deprecated in Python 3.12
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


MODEL_NAME = "{{ cookiecutter.project_name }}"
MODEL_VERSION = "0.0.1"
USE_SERVING_RUNTIME = _strtobool(getenv("USE_SERVING_RUNTIME") or "False")

CACHE_TTL = 600