
from sklearn.metrics import precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression

import numpy as np
import inspect

from loguru import logger


class CallibratedClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrapper that modifies a base classifier to return expected precision, variance output and calibrated probability in addition to raw probability.
    Also allows to use validation set for early stopping or overfitting tracking.

    Usage:
    - Use predict or predict_proba for tuning and model comparison using cross-validation.
    - Train the model as usual using a training set.
    - Calibrate the model with refit=False on a held-out calibration set (it will "learn" the mapping).
    - Use predict_expected_precision in production to flag users 
    
    DON'T:
        The expected precision is not the model's true test precision — it's "what precision looked like for a similar score in the calibration set."
        That is a data leakage path if used for evaluation, because the calibration reflects how good the model was on known data.
        -> SHOULD NOT BE USED AS A SCORING PERFORMANCE (example for hyperparameter tuning or to compare model)
    """
    
    def __init__(self, base_estimator=None):
        
        self.base_estimator = base_estimator # underlying model
        
        self.X_calib = None
        self.y_calib = None
        
        self.prec = None # list of precision for given thresholds
        self.op = None # list of thresholds
        
        self.use_beta = None

        self.is_fitted_ = False

    def fit(self, X_calib, y_calib, refit=True, X_val=None, y_val=None, patience=50, tolerance=0.001):
        """ 
        fit/train (refit=True) or calibrate (refit=False) the base estimator. 
        If refit is false -> calibration: learns the calibration mapping and the proba to precision maping
        If refit is True -> learned maping comes from the training set (data leakage)

        X_val, y_val -> validation set for early stopping
        patience (int): minimum number of optimzation round to wait before applying early stopping criteria on loss. Default to 20 (too large for practical purpose)
        tolerance (float): tolerance in loss for early stopping criteria.
        """
        
        # check if given base_estimator accepts validation set as input (only for PyTorch and XGboost models)
        fit_sig = inspect.signature(self.base_estimator.fit)
        allowed_keys = fit_sig.parameters.keys()
        provided_kwargs = {"X_val": X_val, "y_val": y_val, "patience":patience, "tolerance":tolerance}
        fit_kwargs = {k: v for k, v in provided_kwargs.items() if k in allowed_keys}

        # Log a warning if val data is provided but will be ignored
        if (X_val is not None or y_val is not None) and not {"X_val", "y_val"} <= allowed_keys:
            logger.warning(
                f"Validation data (X_val, y_val) was provided but is being ignored: "
                f"{self.base_estimator.__class__.__name__} does not support it."
            )

        # store training/calibration set
        self.X_calib = X_calib
        self.y_calib = y_calib

        # refit if training mode
        if refit:
            if self.is_fitted_:
                logger.info("Re-Train classifier")
            else:
                logger.info("Train classifier")
            self.base_estimator.fit(X_calib, y_calib, **fit_kwargs)
        else:
            if self.is_fitted_:
                logger.info("Calibrate classifier")
            else:
                logger.error("Model must be trained before calibration")

        # need to have classes_ attribute for sklearn compatibility
        if hasattr(self.base_estimator, "classes_"):
            self.classes_ = self.base_estimator.classes_
        else:
            raise ValueError(f"base estimator {self.base_estimator} must have the attribute 'classes_'")
        
        self.use_beta = self.base_estimator.use_beta if hasattr(self.base_estimator, "use_beta") else None

        # get probability from estimator
        y_proba = np.array(self.base_estimator.predict_proba(self.X_calib), ndmin=2)[:, 1]

        # get the precision-recall curve for expected precision maping
        self.prec, _, self.op = precision_recall_curve(self.y_calib, y_proba)

        # use isotonic regression for probability calibration
        self._isotonic = IsotonicRegression(out_of_bounds="clip", y_max=1, y_min=0)
        self._isotonic.fit(y_proba, self.y_calib)
        
        self.is_fitted_ = True # for sklearn compatibility

        #return self

    def _expected_precision(self, p):
        """
        Take the precision_recall_curve obtain at fit, interpolate it and return expected precision for given probability used as a threshold.
        
        Example:
            In production, if one user is "flagged" with a proba of 0.83, we would have:
            “This user has a 0.83 probability of being a scammer, according to the model” 
            BUT
            “Based on calibration data, when we use a threshold of 0.83 for the model raw output probability, the actual precision was 0.68. So we will return 0.68 instead.”

        Args:
            p (array): raw prediction probability

        Returns:
            1D array: expected precision for threshold = p
        """
        return np.interp(p, self.op, self.prec[:-1])
    
    def _calibrated_isotonic(self, p):
        """
        Take the isotonic maping at fit and return expected probability given raw probability output

        Args:
            p (array): raw prediction probability

        Returns:
            1D array: expected probability P(y=1|p) 
        """
        return self._isotonic.predict(p)

    def predict_expected_precision(self, X):
        """
        The function returns maped predicted probabilities into expected precision values — based on how the model performs on a calibration set.

        Args:
            X (pd.DataFrame): features dataframe.

        Returns:
            1D array: expected precision
        """
        p = np.array(self.base_estimator.predict_proba(X), ndmin=2)[:, 1]
        return self._expected_precision(p)
    
    def predict_calibrated(self, X):
        """
        The function returns maped probabilities into calibrated probabilities — based on how the model performs on a calibration set.

        Args:
            X (pd.DataFrame): features dataframe.
        
        Returns:
            1D array: calibrated probability
        """
        p = np.array(self.base_estimator.predict_proba(X), ndmin=2)[:, 1]
        return self._calibrated_isotonic(p)

    def predict(self, X):
        """
        call base estimator predict method
        
        Returns:
            2D array of flag
        """
        return self.base_estimator.predict(X) 

    def predict_proba(self, X):
        """ 
        call base estimator predict_proba method
        
        Returns:

            2D array raw probability (class 0, class 1)
        """
        return self.base_estimator.predict_proba(X)
    
    def predict_uncert(self, X):
        """
        If the Model is uncertainity-aware, return the mean and variance of the Beta distribution for each data points

        Args:
            X (array-like): input features.

        Returns:
            pd.DataFrame: dataframe with X.index as index and columns 'mu' and 'var'
        """
        if self.use_beta is None:
            logger.warning("Attempting to get mean and variance values from model that does not allow uncertainity-aware prediction. Returning raw output probability.")
            return self.predict_proba(X)
        elif not self.base_estimator.use_beta:
            logger.error("Model has not be trained in uncertainty-aware mode. Re-run the model with use_beta=True")
        else:
            return self.base_estimator.predict_uncert(X)

    def _repr_html_(self):
        """ 
        Use base estimator's HTML for tree like visual in notebook, if available
        """
        if hasattr(self.base_estimator, "_repr_html_"):
            return self.base_estimator._repr_html_()
        else:
            return f"<pre>{repr(self)}</pre>"