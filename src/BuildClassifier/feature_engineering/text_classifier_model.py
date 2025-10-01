from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid parralelization warning from huggingface
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from collections import Counter
from itertools import islice

# import decorators
from BuildClassifier.feature_engineering.decorators import *

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from loguru import logger

# TODO: add a class for processing the sentence embedding using clustering such as kmeans and return categories instead of RF probability output
# TODO: add a class for processing the text using bow appraoch instead of vector embeddings (better suited to larger text)


class Embeddings:

    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2'):
        # other possible model: "hkunlp/instructor-large"
        logger.info(f"Initializing {model} model")
        self.model_embeddings = SentenceTransformer(model)
    
    def get_embendings(self, text, llm_context=""):
        """ get sentence embedding vector for a list of free text with additional context"""
        text_to_embend = [llm_context+t for t in text]
        emb = self.model_embeddings.encode(text_to_embend)
        logger.success("Free text field embedded")
        return emb

class RandomForestEmbed(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that trains a Random Forest on embeddings
    and returns the predicted probabilities as features.

    Parameters:
    -----------
    n_estimators: int
        Number of trees in the Random Forest.
    max_depth: int or None
        Maximum depth of the trees.
    random_state: int
        Random seed for reproducibility.
    """
    
    # Note: only 5 parameters of the RF can be defined: n_estimator, max_depth, min_samples_split, min_samples_leaf and n_jobs
    # TODO: allow full parameters definition via kwargs -> need to define set_params and get_params function for sklearn compatibility (see XGBoost_wraper module)

    def __init__(self, n_estimators=100, max_depth=20, min_samples_split=2, min_samples_leaf=1, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_jobs=n_jobs
        self.rf_model = None  # Placeholder for the trained RF model
    
    def set_output(self, transform="pandas"):
        # for sklearn transformation part of the pipeline compatibility
        return self

    def fit(self, X:pd.DataFrame, y):
        """
        Trains a Random Forest on the embedding vectors.

        Parameters:
        -----------
        X: X: DataFrame containing one column with the Feature matrix (embedding vectors) as array.
        y: np.ndarray. Target labels.

        Returns:
        --------
        self
        """

        if len(np.unique(y)) == 1:
            logger.error(f"RandomForestEmbed: y has only one class: {np.unique(y)}")
            raise ValueError(f"RandomForestEmbed: y has only one class: {np.unique(y)}")

        X_values, col_name = self._convert_df(X)

        logger.info(f"{X.columns[0]}: Text classifier fit: start")

        self.rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split, 
            min_samples_leaf=self.min_samples_leaf, 
            n_jobs=self.n_jobs
        )
        self.rf_model.fit(X_values, y)
        logger.success(f"{col_name}: Text classifier fit: done")

        self.is_fitted_ = True # for skleran compatibility
        return self

    def transform(self, X:pd.DataFrame):
        """
        Returns the predicted probabilities from the trained RF model.

        Parameters:
        -----------
        X: DataFrame containing one column with the Feature matrix (embedding vectors) as array.

        Returns:
        --------
        np.ndarray
            Predicted probabilities for the positive class.
        """
        X_values, col_name = self._convert_df(X)

        logger.info(f"{col_name}: Text classifier transform: start")
        
        if self.rf_model is None:
            err_str = "RandomForestEmbed has not been fitted yet. Call `fit()` first."
            logger.error(err_str)
            raise ValueError(err_str)

        probs = self.rf_model.predict_proba(X_values)[:, 1].reshape(-1, 1)  # Returning only the positive class prob.

        logger.success(f"{col_name}: Text classifier transform: done")
        
        # Return a DataFrame with one column
        return pd.DataFrame(probs, columns=[col_name], index=X.index)    
        
    def fit_transform(self, X, y):
        """
        Convenience method that fits and transforms in one step.

        Parameters:
        -----------
        X: DataFrame containing one column with the Feature matrix (embedding vectors) as array.
        y: np.ndarray
            Target labels.

        Returns:
        --------
        np.ndarray
            Predicted probabilities for the positive class.
        """
        return self.fit(X, y).transform(X)
    
    def _convert_df(self, X:pd.DataFrame):
        
        # check if dataframe has the correct format
        if not isinstance(X, pd.DataFrame):
            err_str = "Input must be a dataframe"
            logger.error(err_str)
            raise ValueError(err_str)
        if len(X.columns) != 1:
            err_str = "Input dataframe must contain only one column"
            logger.error(err_str)
            raise ValueError(err_str)
        
        # extract column values
        X_col = X.iloc[:,0]
        col_name = X_col.name
        X_values = np.vstack(X_col.values) # unstack embedding vectors

        return X_values, col_name

def create_embeddings_features(str_series:pd.Series, llm_context:str, model='sentence-transformers/all-MiniLM-L6-v2'):
    """ helper function to get vectore embeddings """
    emb_class = Embeddings(model=model)
    # Generate embeddings for all text data with additional context
    emb_mat = emb_class.get_embendings(str_series.astype(str), llm_context=llm_context)
    return [row for row in emb_mat]

def embedding_rfprob_features(df, col_name, llm_context, flag_col_name="scam_flag", n_folds=3, rf_para={}):
    """
    Applies a Random Forest on Sentence Embeddings with K-Fold Training 
    to prevent target leakage.
    To be used as testing, outside of the pipeline

    Parameters:
    df: pd.DataFrame -> Dataset containing text data.
    col_name: str -> Column name containing the text.
    llm_context: str -> Context for the embedding model.
    flag_col_name: str -> Target column name (default: "scam_flag").
    n_folds: int -> Number of K-Folds (default: 3).

    Returns:
    tuple -> (Out-of-Fold Predictions, Final Model for Test Data, Classification Report)
    """
    
    tmp_df = df.copy()
    
    emb_class = Embeddings()
    
    # Generate embeddings for all text data
    embeddings = emb_class.get_embendings(tmp_df[col_name].astype(str), llm_context=llm_context)
    
    # Prepare out-of-fold storage
    oof_predictions = np.zeros(len(tmp_df))
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    for train_idx, val_idx in kf.split(tmp_df):
        train_embeddings, val_embeddings = embeddings[train_idx], embeddings[val_idx]
        train_labels, val_labels = tmp_df.iloc[train_idx][flag_col_name], tmp_df.iloc[val_idx][flag_col_name]
        
        # Train RF on (K-1) folds
        rf_classifier = RandomForestClassifier(**rf_para)
        rf_classifier.fit(train_embeddings, train_labels)
        
        # Predict on the held-out fold
        oof_predictions[val_idx] = rf_classifier.predict_proba(val_embeddings)[:, 1]
    
    # Train final model on full dataset (for test use only)
    final_rf_classifier = RandomForestClassifier()
    final_rf_classifier.fit(embeddings, tmp_df[flag_col_name])
    
    # Evaluate OOF Predictions
    report = classification_report(oof_predictions.round(), tmp_df[flag_col_name])
    
    return oof_predictions, final_rf_classifier, report