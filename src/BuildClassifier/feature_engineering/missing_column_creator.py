import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from loguru import logger


class MissingCountIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, output_column='missing_count'):
        self.columns = columns
        self.output_column = output_column

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.columns_ = self.columns if self.columns is not None else X.columns.tolist()

        # Add the missing count column now, so later transformers (e.g., scaler, NN) see consistent shape
        self.missing_count_ = X[self.columns_].isna().sum(axis=1)
        self.feature_names_out_ = X.columns.tolist() + [self.output_column]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X[self.output_column] = X[self.columns_].isna().sum(axis=1)
        logger.success(f"Added missingness columns: new shape = {X.shape}")
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_



