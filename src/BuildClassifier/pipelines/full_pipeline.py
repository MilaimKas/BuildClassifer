"""
Module containing the get_model_pipeline helper function for creating the sklearn pipeline object.
"""

import numpy as np
import copy

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

from joblib import Memory

# Feature engineering and selection
from category_encoders.target_encoder import TargetEncoder
from category_encoders.count import CountEncoder

from BuildClassifier.feature_engineering.text_classifier_model import RandomForestEmbed
from BuildClassifier.feature_engineering.feature_transformation_utilities import ClipTransformer
from BuildClassifier.feature_engineering.missing_column_creator import MissingCountIndicator
from BuildClassifier.feature_selection.feature_selector import ColumnSelector
from BuildClassifier.pipelines.pipeline_parameters import *

# Models
from BuildClassifier.models.Pytorch_model_wrapper import TorchClassifier
from BuildClassifier.models.calibration_wrapper import CallibratedClassifier
from BuildClassifier.models.Tree_model_wrapper import XGBoostClassifierWrapper, UncertaintyAwareRF

# Logging and validation
from BuildClassifier.pipelines.logger_wrapper import LoggingTransformer, AssertNoCategoricals
from loguru import logger

from BuildClassifier.pipelines.pipeline_helpers import *


def get_model_pipeline(model="pytorch", cache=True, caching_dir="caching/", 
                        feature_creation_class=None, 
                        free_text_features=[],
                        cat_to_targetencode_A=[],
                        cat_to_targetencode_B=[],
                        cat_to_targetencode_C=[],
                        cat_to_count=[],
                        num_to_clip_sym=[],
                        num_to_clip_upper=[],
                        num_to_logtransform=[],
                        num_to_normalize=[],
                        selected_features=[],
                        trial=None,
                        classifier_kwargs={}):
    """
    Build a full sklearn pipeline with feature transformation and classification.

    Args:
        model (str or BaseEstimator): "pytorch", "randomforest", or a custom estimator. Note: the estimators are wrapped inside a Callibrator class.
        cache (bool): whether to cache transformation steps.
        caching_dir (str): path for joblib caching.
        feature_creation_class (base_estimator): a base_estimator FeatureCreator() class to be included in the pipeline (simplify inference mode). Defaut to None.
        free_text_features (list): list of free text features to handle using RF text classifier.
        cat_to_targetencode_A (list): list of categorical features with low cardinality/skewdness.
        cat_to_targetencode_B (list): list of categorical features with medium cardinality/skewdness.
        cat_to_targetencode_C (list): list of categorical features with high cardinality/skewdness.
        cat_to_count (list): list of categories to encode using count.
        num_to_clip_sym (list): list of numerical features to clip from both side.
        num_to_clip_upper (list): list of numerical features to clip (upper bound only).
        num_to_logtransform (list): list of numerical features to logtransform.
        num_to_normalize (list): listf or numerical features to normalized (usually all original numerical features).
        selected_features (list): list of features to keep.
        trial (optuna.trial): use when performing optuna tuning, will triggerd pruning if implemented.
        classifier_kwargs (dict): arguments for the classifier object.

    Returns:
        sklearn.pipeline.Pipeline: complete pipeline object.
    """

    # available models
    model_dict = {
        "pytorch": TorchClassifier,
        "randomforest": UncertaintyAwareRF,
        "xgboost": XGBoostClassifierWrapper,
    }

    # check model input and initialize classifier using ExpectedPrecision wrapper
    if isinstance(model, str):
        if model not in model_dict:
            err_str = f"Invalid model name, must be any of {model_dict.keys()}, got {model}"    
            logger.error(err_str)
            raise ValueError(err_str)
        if model != "randomforest": # no trial pruning implemented for RandomForest
            classifier = CallibratedClassifier(model_dict[model](trial=trial, **classifier_kwargs))
        else:
            classifier = CallibratedClassifier(model_dict[model](**classifier_kwargs))
    # check if Base estimator given as input. Wrap in ExpectedPrecision
    elif isinstance(model, BaseEstimator):
        classifier = CallibratedClassifier(model)
    else:
        raise ValueError(f"Invalid model: expected str or Base Estimator, got {type(model)}")

    # use pipeline caching
    memory = Memory(caching_dir, verbose=0) if cache else None

    # kwargs for ColumnTransformer
    col_trans_kwargs = {"remainder":"passthrough", "verbose_feature_names_out":False, "sparse_threshold":0, "force_int_remainder_cols":False}

    # Filter feature groups if not found in selected_features
    def filt(lst): return [f for f in lst if f in selected_features] if selected_features else lst
    num_to_clip_sym_f, num_to_clip_upper_f = filt(num_to_clip_sym), filt(num_to_clip_upper)
    free_text_f, cat_A_f, cat_B_f, cat_C_f = filt(free_text_features), filt(cat_to_targetencode_A), filt(cat_to_targetencode_B), filt(cat_to_targetencode_C)
    cat_count_f, num_log_f, num_to_normalize_f = filt(cat_to_count), filt(num_to_logtransform), filt(num_to_normalize)

    logger.info(f"Creating pipeline with following feature groups: \n\
                - free_text_features text: {free_text_f} \n\
                - cat_to_targetencode_A: {cat_A_f} \n\
                - cat_to_targetencode_B: {cat_B_f} \n\
                - cat_to_targetencode_C: {cat_C_f} \n\
                - num_to_clip_sym: {num_to_clip_sym_f} \n\
                - num_to_clip_upper: {num_to_clip_upper_f} \n\
                - cat_to_count: {cat_count_f} \n\
                - num_to_logtransform: {num_log_f} \n\
                - num_to_normalize: {num_to_normalize_f} \n"
                )

    # initialize pipeline
    pipeline = []

    # if feature creation part of the pipeline (should not be included for tuning but for prediction mode)
    if feature_creation_class:
        if not isinstance(feature_creation_class, BaseEstimator):
            err_str = "Given FeatureCreation class is not a base estimator (does not contain fit and transform methods)"
            logger.error(err_str)
            raise ValueError(err_str)
        pipeline.append(('feature_creation', feature_creation_class))

    # add pipeline steps
    pipeline += [
        ("Free_text_embedding_model", safe_column_transformer([
            (f'rf_{col}', RandomForestEmbed(), [col]) for col in free_text_f
            ], **col_trans_kwargs)),

        ('logging_categorical_encoding', LoggingTransformer("Categorical features encoding")),
        ('categorical_encoding', safe_column_transformer([
            ('te_A', TargetEncoder(), cat_A_f),
            ('te_B', TargetEncoder(), cat_B_f),
            ('te_C', TargetEncoder(), cat_C_f),
            ('count_enc', CountEncoder(min_group_name="leftover", min_group_size=0.01), cat_count_f),
        ], **col_trans_kwargs)),

        ('logging_missingness_indicator', LoggingTransformer("Add missingness count column")),
        ('missingness_indicator', MissingCountIndicator(columns=None)),

        ('feature_selection', ColumnSelector(selected_features)),
        ('Checking_categorical_features', AssertNoCategoricals()),

        ('logging_clipping_transform', LoggingTransformer("Clipping")),
        ('Clipping_transformation', safe_column_transformer([
            ("clipping_sym", ClipTransformer(lower=0.01, upper=0.99), num_to_clip_sym_f),
            ("clipping_upper", ClipTransformer(lower=0, upper=0.99), num_to_clip_upper_f),
        ], **col_trans_kwargs)),

        ('logging_log_transform', LoggingTransformer("Logarithm")),
        ('Log_transformation', safe_column_transformer([
            ("log_transform", FunctionTransformer(np.log1p), num_log_f)
        ], **col_trans_kwargs)),

        ('logging_standardization', LoggingTransformer("Inputing + scaling", print_dtype=True)),
        ('replace_nan', SimpleImputer(missing_values=np.nan, strategy='median').set_output(transform="pandas")),
        ('feature_standardization', safe_column_transformer([
            ('normalisation', StandardScaler(), num_to_normalize_f)
        ], **col_trans_kwargs)),

        ('logging_classifier', LoggingTransformer("Model step")),
        ('classifier', classifier)
    ]

    return Pipeline(pipeline, memory=memory)


def build_pipeline(params, model_name, cache=False, pipeline_kwargs={}):
    """
    Utility to build pipeline from parameter dict output from optuna.

    Args:
        params (dict): full parameter dictionary where key=pipeline parameter name and value=parameter value, with selected features. 
        model_name (str): one of ["pytorch", "randomforest", "skorch"].
        trial (optuna.Trial): passed to TorchClassifier.
        cache (bool): toggle caching.

    Returns:
        sklearn.Pipeline: complete pipeline.
    """

    if not isinstance(params, dict):
        raise ValueError("params input must be a dictionary with pipeline parameters as key")

    # check selected feature key
    if "selected_features" not in params.keys():
        raise ValueError("'selected_features' must be provided in params")

    selected_features = params["selected_features"]
    pipeline_params = {k: v for k, v in params.items() if k not in ["selected_features", "selected_feature_indices", "top_k"]}

    pipe = get_model_pipeline(model=model_name, selected_features=selected_features, cache=cache, **pipeline_kwargs)

    # filtered out invalid parameters
    params_filtered = filter_valid_params(pipe, params)

    return pipe.set_params(**params_filtered)



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from loguru import logger
    import sys
    #import random, string

    # add only informative logs
    logger.remove()
    logger.add(sys.stderr, level="SUCCESS")
    logger.add(sys.stderr, level="WARNING")
    logger.add(sys.stderr, level="ERROR")

    # Fake data
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"col_{i}"for i in range(10)])
    # add random free text
    #X_df["text"] = [''.join(random.sample(string.ascii_letters, random.randint(2, 10))) for i in range(len(X))]
    # add random embeddings
    X_df["text"] = [np.random.uniform(low=0, high=13.3, size=(50,)) for i in range(len(X))]

    # add random categorical feature
    X_df["categorical"] = np.random.choice(["a", "b", "c", "d"], len(X))

    # split train-test
    X_train, X_val, y_train, y_val = train_test_split(X_df, y, test_size=0.2)

    # Dummy selected features
    selected_features = ["text", "categorical", "col_1", "col_2", "col_5", "col_7", "col_8"]

    # feature type definition
    pipeline_kwargs={"cat_to_targetencode_A": ["categorical"],
                    "num_to_clip_upper":["col_1", "col_2"], 
                    "free_text_features":["text"], 
                    "num_to_normalize":["col_1", "col_2", "col_5", "col_7", "col_8"]}

    # default parameters
    params = {'classifier__base_estimator__epochs': 49,
        'classifier__base_estimator__dropout': 0.07,
        'classifier__base_estimator__lr': 7e-05,
        'classifier__base_estimator__batch_size': 100,
        'classifier__base_estimator__hidden_activation': 'tanh',
        'classifier__base_estimator__dense': '160_192_256',
        'Free_text_embedding_model__rf_text__n_estimators': 20,
        'Free_text_embedding_model__rf_text__max_depth': 30,
        'categorical_encoding__te_A__smoothing': 5
        }
    params.update({"selected_features":selected_features})

    # Build pipeline
    pipeline = get_model_pipeline(model="randomforest", selected_features=selected_features, cache=False)
    print("Pipeline without transformation steps created successfully.")
    pipeline = build_pipeline(params, "pytorch", cache=False, pipeline_kwargs=pipeline_kwargs)
    print("Pipeline with input paramaters created successfully using helper.")

    # simulate training 
    pipeline.fit(X_train, y_train)
    print("Pipeline succesfully trained")

    # simulate prediction (raw output probability, calibrated probability and exptected precision)
    pred_proba = pipeline.predict_proba(X_val)
    pred = pipeline.predict(X_val)
    exp_pre = predict_expected_precision(pipeline, X_val)
    cal_pre = predict_calibrated(pipeline, X_val)
    print("Pipeline prediction succesfully")

    # Simulate training with validation monitoring
    try:
        train_with_val(pipeline, X_train, y_train, X_val, y_val)
        print("Pipeline succesfully trained with validation monitoring.")
    except Exception as e:
        print(f"Training failed: {e}")
    
    print("Check validation curve")
    
    
    print()
    print("################################")
    print("Predict using uncertainity model")
    print("################################")
    print()

    # check uncertainity output
    params.update({"classifier__base_estimator__use_beta":True})
    
    pipeline = build_pipeline(params, "randomforest", cache=False, pipeline_kwargs=pipeline_kwargs)
    print("Pipeline with uncertainity-aware randomforest model created successfully.")
    
    pipeline.fit(X_train, y_train)
    print("Pipeline succesfully trained")

    X_val_transformed = pipeline[:-1].transform(X_val)
    uncert_df = pipeline.named_steps["classifier"].predict_uncert(X_val_transformed)
    print(uncert_df.head(10))
