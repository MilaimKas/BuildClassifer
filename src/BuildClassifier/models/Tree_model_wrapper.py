
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier, callback as xgb_callback
from xgboost import DMatrix

from optuna.integration import XGBoostPruningCallback
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

from scipy.stats import beta

from loguru import logger


class XGBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for XGBoost with Optuna pruning and early stopping support.

    Parameters
    ----------
    trial : optuna.Trial, optional
        Current Optuna trial object. Used for pruning.
    n_estimators : int, optional
        Maximum number of boosting rounds.
    early_stopping_rounds : int, optional
        Enables early stopping based on eval set performance.
    eval_metric : str, optional
        Evaluation metric used by XGBoost.
    xgb_params : dict
        Parameters to be passed to XGBClassifier constructor.
    """

    def __init__(self, trial=None, n_estimators=500, eval_metric="logloss", **xgb_params):
        self.trial = trial
        self.n_estimators = n_estimators
        self.eval_metric = eval_metric
        self.xgb_params = xgb_params
        self.model = None

         # need for sklearn compatibility: hard coded binary classification variable
        self.classes_ = np.array([0, 1])

        # store train an validation loss
        self.train_loss = None
        self.val_loss = None

        self.patience = None
        self.tolerance = None

    def fit(self, X, y, X_val=None, y_val=None, patience=20, tolerance=0.001):
        """Train the XGBoost model with optional validation set and pruning."""

        self.patience = patience
        self.tolerance = tolerance

        eval_set = [(X, y)] # define evaluation set with training data
        callbacks = []

        # check if attribute trial is not None (Optuna compatibility)
        trial = getattr(self, "trial", None)
        if trial is not None:
            logger.info("Checking for pruning optuna's trial")
            if self.trial:
                callbacks.append(XGBoostPruningCallback(self.trial, f"validation_0-{self.eval_metric}"))
        
        if X_val is not None and y_val is not None and patience and tolerance:
            eval_set.append((X_val, y_val)) # add validation data to the evaluation sets
            logger.info(f"XGBoost will use validation set for early stoping with patience={patience} and tolerance={tolerance}")
            # name of the second data set is validation_1 (validation_0 is the training data)
            callbacks.append(xgb_callback.EarlyStopping(rounds=patience, data_name="validation_1", tolerance=tolerance))

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            eval_metric=self.eval_metric,
            use_label_encoder=False,
            objective="binary:logistic",
            tree_method="hist",
            callbacks=callbacks,
            verbosity=0,
            **self.xgb_params
        )

        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False
        )

        # store train and val loss
        if X_val is not None and y_val is not None and patience:
            evals_result = self.model.evals_result()
            self.train_loss = evals_result["validation_0"][self.eval_metric]
            self.val_loss = evals_result["validation_1"][self.eval_metric]

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict_uncert(self, X):
        logger.warning("Uncertainity output is not yet implemented for XGBoost models. Use a RF model insetad. Returning raw output probability and zeros variance.")
        return pd.DataFrame({'mu': self.predict_proba(X)[:,1], 'var': np.zeros(len(X))}, index=X.index)


    def get_params(self, deep=True):
        """ 
        This overwrites the native BaseEstimator.get_params() function. 
        Required here for sklearn compatibility because of the kwargs in __init__
        """
        out = dict(
            trial=self.trial,
            n_estimators=self.n_estimators,
            patience=self.patience,
            tolerance=self.tolerance,
            eval_metric=self.eval_metric,
            **self.xgb_params
        )
        return out

    def set_params(self, **params):
        """ 
        This overwrites the native BaseEstimator.set_params() function. 
        Required here for sklearn compatibility because of the kwargs in __init__
        """
        for key, val in params.items():
            if key in {"trial", "n_estimators", "early_stopping_rounds", "eval_metric"}:
                setattr(self, key, val)
            else:
                self.xgb_params[key] = val
        return self


class UncertaintyAwareRF(RandomForestClassifier):
    """
    A drop-in replacement for sklearn's RandomForestClassifier with uncertainty-aware inference
    using Beta distributions estimated from per-leaf label statistics.
    """

    def __init__(self, use_beta=False, **kwargs):
        super().__init__(**kwargs)
        self._leaf_stats = None
        self.use_beta = use_beta

    def fit(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        super().fit(X, y)
        if self.use_beta:
            self._build_leaf_statistics(X, y)
        return self

    def _build_leaf_statistics(self, X_train, y_train):
        """Cache per‑leaf Beta parameters (k+1, n−k+1) for every tree."""
        self._leaf_stats = []
        y_series = pd.Series(y_train)
        # loop over trees
        for tree in self.estimators_:
            # find idx of the leaf where split was made
            leaf_ids = tree.apply(X_train.to_numpy())
            stats = {}
            # vectorised aggregation
            unique_leafs, inverse = np.unique(leaf_ids, return_inverse=True)
            counts = np.bincount(inverse)
            sums = np.bincount(inverse, weights=y_series.values)
            for leaf, n, k in zip(unique_leafs, counts, sums):
                stats[leaf] = (k + 1, n - k + 1)  # (alpha, beta)
            self._leaf_stats.append(stats)

    def _per_tree_beta_params(self, X: pd.DataFrame):
        """
        Return two arrays [n_trees, n_samples] with the per-tree alpha and beta
        parameters for all samples in *X*.
        """
        if self._leaf_stats is None:
            raise RuntimeError("Call .fit(..., use_beta=True) first.")

        alpha_list, beta_list = [], []

        X_np = X.to_numpy()
        for stats, tree in zip(self._leaf_stats, self.estimators_):
            leaf_ids = tree.apply(X_np)                      # shape (n_samples,)
            # look up (alpha, beta) for each leaf; default (1,1) if unseen
            a_b = np.array([stats.get(lid, (1, 1)) for lid in leaf_ids])
            alpha_list.append(a_b[:, 0])
            beta_list.append(a_b[:, 1])

        return np.vstack(alpha_list), np.vstack(beta_list)

    def predict_uncert(self, X: pd.DataFrame):
        """Predict class‑1 probability *μ* and variance *σ²*.
        Returns also the epistemic and aleatoric part of the uncertainty
        """

        if not self.use_beta:
            err_str = "Attempting to get mean and variance from a RandomForest model that is not uncertainty-aware. \
                        Re-define the model with the argument 'use_beta=True'"
            logger.error(err_str)
            raise ValueError(err_str)

        # get alpha nand beta parameters for each trees
        alpha, beta = self._per_tree_beta_params(X)
        
        # get mean and var of the Beta distribution for each tree
        mu_t = alpha / (alpha + beta)                       
        var_t = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        # aggregate mean over trees
        mu = mu_t.mean(axis=0)
        # aleatoric uncertainity as mean of tree variances
        alea = var_t.mean(axis=0)      
        # epistemic uncertainty as variance of means                        
        epi = mu_t.var(axis=0, ddof=1)    
        # total uncertainty as sum                     
        var = alea + epi
        return pd.DataFrame({'mu': mu, 'var': var, 'alea':alea, "epi":epi}, index=X.index)

    def get_params(self, deep=True):
        # compatibility with sklearn pipeline
        return super().get_params(deep=deep)

    def set_params(self, **params):
        # compatibility with sklearn pipeline
        super().set_params(**params)
        return self




if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Now test both models
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"col_{i}"for i in range(5)])
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    rf_model = UncertaintyAwareRF(n_estimators=100, max_depth=4, random_state=42, use_beta=True)
    rf_model.fit(X_train, y_train)

    xgb_model = XGBoostClassifierWrapper(n_estimators=100, max_depth=4, random_state=42)
    xgb_model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    x = X_test.iloc[0:10,:]
    
    Uncert_df_rf = rf_model.predict_uncert(x)
    y_proba_rf = rf_model.predict_proba(x)[:,1]

    y_proba_xg = xgb_model.predict_proba(x)[:,1]

    print()
    print("Uncertainty Aware Random Forest output")
    print(Uncert_df_rf)
    print("MAE = ", round(np.mean(np.abs(Uncert_df_rf["mu"]-y_proba_rf))*100, 1))

    print()
    print("XGBoost probability output")
    print(y_proba_xg)
    print()
    print("XGBoost train Vs validation loss")
    df_loss = pd.DataFrame()
    df_loss["val_loss"] = xgb_model.val_loss
    df_loss["training_loss"] = xgb_model.train_loss
    print(df_loss)
    print()

