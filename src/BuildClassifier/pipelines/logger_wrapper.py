from sklearn.base import TransformerMixin, BaseEstimator
from loguru import logger
import pandas as pd
import numpy as np
import textwrap

class LoggingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to use inside a pipeline for logging/debugging and sanity checks
    """

    def __init__(self, step_name="", print_dtype=False, display_columns=False):
        self.step_name = step_name
        self.print_dtype = print_dtype
        self.display_columns = display_columns
        self.list_of_features_at_fit = None

    def fit(self, X, y):
        self._log_info("fit", X)
        if hasattr(X, 'columns'):
            self.list_of_features_at_fit = list(X.columns)
        logger.debug(f"Target shape: {y.shape} - Target dtype: {y.dtype}")
        self.is_fitted_ = True # for skleran compatibility
        return self

    def transform(self, X):
        self._log_info("transform", X)
        self._check_feature_mismatch(X)
        return X

    def _log_info(self, stage, X):
        if (self.print_dtype) & (hasattr(X, 'dtypes')):
            data_type = f", data-types={np.unique(X.dtypes)}"
        else:
            data_type = ""
        if hasattr(X, 'shape') and hasattr(X, 'columns'):
            if self.display_columns:
                cols = "columns= {}".format(list(X.columns))
            else:
                cols = ""
            feature_str = ", ".join(list(X.columns))
            wrapped_feature_str = textwrap.fill(feature_str, width=100)  # wrap at 100 characters
            logger.debug(f"🔹 {self.step_name} - {stage}: shape={X.shape}, {cols} {data_type}")
            logger.debug(f"🔹 {self.step_name} - {stage}: features=\n{wrapped_feature_str}")
        elif hasattr(X, 'shape'):
            logger.debug(f"🔹 {self.step_name} - {stage}: shape={X.shape} {data_type}, type={type(X).__name__}")
        else:
            logger.debug(f"🔹 {self.step_name} - {stage}: type={type(X).__name__} {data_type}")
    
    def set_output(self, transform=None):
        """ compatibility with sklearn pipeline """
        return self
    
    def _check_feature_mismatch(self, X):
        if hasattr(X, 'columns'):
            current_features = list(X.columns)
            if not hasattr(self, 'list_of_features_at_fit'):
                logger.warning(f"Transformer {self.step_name} not fitted.")
            else:
                if current_features != self.list_of_features_at_fit:
                    logger.error(f"[LoggingTransformer] ❌ Mismatch in feature columns:\n"
                                f"At fit:      {len(self.list_of_features_at_fit)} features\n"
                                f"At transform:{len(current_features)} features\n"
                                f"Missing at transform: {set(self.list_of_features_at_fit) - set(current_features)}\n"
                                f"Unexpected at transform: {set(current_features) - set(self.list_of_features_at_fit)}\n")
                    raise ValueError("Feature mismatch between fit and transform detected.")



class AssertNoCategoricals(BaseEstimator, TransformerMixin):
    """
    Transformer class to add to sklearn pipeline to assess if there are remaining categorical features 
    """
    def fit(self, X, y=None):
        self.is_fitted_ = True # for skleran compatibility
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            logger.debug(f"Checking for remaining categorical variables: shape={X.shape},")
            tmp = X.select_dtypes(include=['object', 'string', 'category'])
            cat_cols = tmp.columns
            cat_type = tmp.dtypes
            if len(cat_cols) > 0:
                cat_cols_dtype = [f"{c}-{d}" for c,d in zip(cat_cols, cat_type)]
                raise ValueError(f"❌ Found un-encoded categorical columns with '[name-dtype]'): {cat_cols_dtype}")
            logger.debug(f"NO remaining categorical variables: shape={X.shape},")
            feature_str = ", ".join(list(X.columns))
            wrapped_feature_str = textwrap.fill(feature_str, width=100)  # wrap at 100 characters
            logger.debug(f"🔹 Categorical check - transform: features=\n{wrapped_feature_str}")
        return X

