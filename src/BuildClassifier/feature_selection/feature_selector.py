from sklearn.base import BaseEstimator
import numpy as np
from pandas import DataFrame, Series
from loguru import logger

class ColumnSelector(BaseEstimator):
    """Object for selecting specific columns from a data set.
    Handles pandas and np.array format.

    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.

    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).
    
    Sources
    ----------
    https://raw.githubusercontent.com/rasbt/mlxtend/master/mlxtend/feature_selection/column_selector.py

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/

    """

    def __init__(self, cols=[], drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit(self, X, y=None):
        self.is_fitted_ = True # for skleran compatibility
        return self

    def transform(self, X, y=None):
        if not self.cols:
            return X

        # Convert tuple to list
        if isinstance(self.cols, tuple):
            self.cols = list(self.cols)

        # Determine input type
        if hasattr(X, "loc") or hasattr(X, "iloc"):
            is_df = True
        else:
            is_df = False

        if is_df:
            missing = [col for col in self.cols if col not in X.columns]
            present = [col for col in self.cols if col in X.columns]

            if missing:
                logger.warning(f"ColumnSelector: Missing columns {missing} will be ignored.")

            if not present:
                logger.error("ColumnSelector: No selected columns found in input. Returning empty array.")
                raise ValueError("ColumnSelector: No selected columns found in input. Returning empty array.")

            t = X.loc[:, present]

        else:  # fallback for NumPy array
            max_idx = X.shape[1]
            present = [i for i in self.cols if i < max_idx]
            missing = [i for i in self.cols if i >= max_idx]

            if missing:
                logger.warning(f"ColumnSelector: Ignoring invalid indices {missing}.")
            if not present:
                logger.error("ColumnSelector: No valid columns found. Returning empty array.")
                raise ValueError("ColumnSelector: No selected columns found in input. Returning empty array.")

            t = X[:, present]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]

        logger.success("Feature selection: done")
        return t

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
        