# https://github.com/osmr/imgclsmob/pull/69

import inspect
import types


def get_config(self):
    """
    This func implements get_config() method
    which is inevitable when save and load models.
    Parameters:
    ----------
    None
    """

    self.config = super(self.__class__, self).get_config()
    self.config.update(self.argdict)
    return self.config


def add_get_config(f):
    """
    This decorator dynamically add get_config() method.
    This must add class.__init__() method
    Parameters:
    ----------
    f : python function, decorate class's __init__() method
    """

    def wrapper(*args, **kwargs):
        signatures = list(inspect.signature(f).parameters.keys())
        argdict = {sig: arg for sig, arg in zip(signatures[1:], args[1:])}
        argdict.update(kwargs)
        object.__setattr__(args[0], "argdict", argdict)

        f(*args, **kwargs)
        # override get_config()
        object.__setattr__(args[0], "get_config", types.MethodType(get_config, args[0]))

    return wrapper
