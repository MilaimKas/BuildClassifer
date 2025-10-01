import numpy as np
import pandas as pd

def validate_column(func):
    """
    Decorator to handle wrong column name
    """
    def wrapper(df, column, *args, **kwargs):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        return func(df, column, *args, **kwargs)
    return wrapper

def handle_exceptions(default=np.nan):
    """
    Decorator that returns 'default' if an exception is raised
    """
    def wrap(f):
        def inner(*args,**kwargs):
            try:
                return f(*args,**kwargs)
            except Exception:
                return default
        return inner
    return wrap