import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from BuildClassifier.feature_engineering.decorators import *

@validate_column
def clip_quantile(df, col:str, clip_type="upper_lower", quantile=[0.05,0.95], log_trans=False) -> np.array:
    """
    Clip the upper 0.95 and lower 0.05 quantile of the distribution 
    """
    lower, upper = df[col].quantile(quantile).values
    if clip_type == "upper_lower":
        return np.clip(df[col], a_min=lower, a_max=upper)
    elif clip_type == "upper":
        return np.clip(df[col], a_min=0, a_max=upper)
    elif clip_type == "lower":
        return np.clip(df[col], a_min=lower, a_max=0)
    else:
        raise ValueError(f"clip_type must be: 'lower', 'upper' or 'upper_lower', got {clip_type}")
    return df_new

class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower)
        self.upper_bounds_ = X_df.quantile(self.upper)
        self.is_fitted_ = True # for skleran compatibility
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
    
    def set_output(self, transform="pandas"):
        return self
