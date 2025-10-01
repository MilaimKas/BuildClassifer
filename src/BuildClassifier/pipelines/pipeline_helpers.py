
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

import re

# Logging and validation
from loguru import logger



# Helper for pipeline training and prediction
# ---------------------------------------------------------------------------------------------------------------

def train_with_val(pipeline, X_train, y_train, X_val, y_val, patience=20, tolerance=0.001):
    """
    Transform data with pipeline and train model with validation monitoring.

    Args:
        pipeline (Pipeline): sklearn pipeline.
        X_train, y_train, X_val, y_val: datasets.
        patience (int): minimum number of optimzation round to wait before applying early stopping criteria on loss. Default to 20 (too large for practical purpose)
        tolerance (float): tolerance in loss for early stopping criteria
    """
    # fit transformation part of the pipeline
    pipeline[:-1].fit(X_train, y_train)
    X_train_proc = pipeline[:-1].transform(X_train)
    X_val_proc = pipeline[:-1].transform(X_val)
    pipeline.named_steps['classifier'].fit(X_train_proc, y_train, X_val=X_val_proc, y_val=y_val, patience=patience, tolerance=tolerance)

def calibrate_pipeline(pipeline, X_calibration, y_calibration):
    """ calibrate pipeline using calibration set """
    X_cal_trans = pipeline[:-1].transform(X_calibration)
    pipeline[-1].fit(X_cal_trans, y_calibration, refit=False)

def predict_expected_precision(pipeline, X, transformed=False):
    """ 
    return expected precision for X. 

    X (pd.DataFrame): data samples
    transformed (bool): if False, X will be transformed using fitted pipeline
    """
    if not transformed:
        x_transformed = pipeline[:-1].transform(X)
    else:
        x_transformed = X
    return pipeline[-1].predict_expected_precision(x_transformed)

def predict_calibrated(pipeline, X, transformed=None):
    """ 
    return calibrated probability for X. 

    X (pd.DataFrame): data samples
    transformed (bool): if False, X will be transformed using fitted pipeline
    """
    if not transformed:
        x_transformed = pipeline[:-1].transform(X)
    else:
        x_transformed = X
    return pipeline[-1].predict_calibrated(x_transformed)

def plot_train_val_loss(model_pipeline):
    """ plot train and validation loss per optimization step """

    base_estimator = model_pipeline[-1].base_estimator

    if not hasattr(base_estimator, 'train_loss'):
        raise ValueError("Base estimator {} does not have validation and training loss.")

    tloss = base_estimator.train_loss
    vloss = base_estimator.val_loss

    plt.plot(tloss, label="training")
    plt.plot(vloss, label="validation")

    plt.xlabel("epochs/optimization steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


# Helper for pipeline creation
# ---------------------------------------------------------------------------------------------------------------


def identity_transform(x):
    """ dummy function for used in safe_column_transformer"""
    return x

def safe_column_transformer(steps, **kwargs):
    """Filter out any transformation steps where the feature list is empty."""
    steps = [(name, trans, cols) for name, trans, cols in steps if cols]
    if not steps:
        try:
            return FunctionTransformer(func=identity_transform, feature_names_out="one-to-one")
        except:
            logger.error("Problem with safe column transformer")

    return ColumnTransformer(steps, **kwargs).set_output(transform="pandas")

def filter_valid_params(pipeline, param_dict):
    """
    Remove parameters from param_dict that do not correspond to any steps in the pipeline.
    """
    valid_params = pipeline.get_params(deep=True)
    return {k: v for k, v in param_dict.items() if k in valid_params}

def check_para(params, pipeline_kwargs):
    """
    Check concordance between given pipeline parameters and given list of feature types.

    Args:
        params (dict): dict of pipeline parameters
        pipeline_kwargs (dict): dict of pipeline kwargs. Can contain list of feature type.

    Raises:
        ValueError: parameter given but no related feature in list
    """
    for p in params.keys():
        if "Free_text" in p:
            feat_name = re.search('rf_(.*)__', p).group(1)
            text_feat_in = pipeline_kwargs.get("free_text_features", None)
            if text_feat_in is None:
                raise ValueError("Free text related parameters given but no text features defined")
            elif feat_name not in text_feat_in:
                raise ValueError(f"Free text parameters defined for feature '{feat_name}' but feature name not present in free text feature list") 
        elif "te_A" in p:
            if pipeline_kwargs.get("cat_to_targetencode_A", None) is None:
                raise ValueError("'Target encoding A' parameters given but no features given in the list.")
        elif "te_B" in p:
            if pipeline_kwargs.get("cat_to_targetencode_B", None) is None:
                raise ValueError("'Target encoding B' parameters given but no features given in the list.")
        elif "te_C" in p:
            if pipeline_kwargs.get("cat_to_targetencode_C", None) is None:
                raise ValueError("'Target encoding C' parameters given but no features given in the list.") 
