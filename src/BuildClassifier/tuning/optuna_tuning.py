import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score

from optuna.importance import get_param_importances
import optuna

import sys
import os

from loguru import logger
import logging

import matplotlib.pyplot as plt

from BuildClassifier.tuning import tuning_helpers
from BuildClassifier.pipelines import full_pipeline, pipeline_parameters, pipeline_helpers


optuna_algorithm_dict = {
    "Grid":optuna.samplers.GridSampler,
    "Random":optuna.samplers.RandomSampler,
    "TPES":optuna.samplers.TPESampler,
    "CMA":optuna.samplers.CmaEsSampler,
    "GPS":optuna.samplers.GPSampler,
    "Partial":optuna.samplers.PartialFixedSampler,
    "Genteic":optuna.samplers.NSGAIISampler,
    "MC":optuna.samplers.QMCSampler
}

def _build_pipeline(params: dict, model_name: str, selected_features: list, cache: bool = False, trial: bool = False, pipeline_kwargs={}):
    """
    Build a full sklearn pipeline (including model) from parameters and selected features.

    Parameters
    ----------
    params : dict
        Dictionary of pipeline parameters (e.g., model hyperparameters, encoding settings).
    model_name : str
        Identifier of model type, one of {"pytorch", "randomforest", etc.}.
    selected_features : list
        List of feature names to include in the pipeline.
    cache : bool, optional
        Whether to enable sklearn `Memory` caching, by default False.
    trial : optuna.Trial or bool
        Trial object for Optuna (used by model wrappers), or False if unused.
    pipeline_kwargs : dict, optional
        Additional arguments for the `get_model_pipeline()` call (e.g., feature type groupings).

    Returns
    -------
    sklearn.Pipeline
        Fully configured pipeline with preprocessing and model steps.
    """
    pipeline_kwargs.update({"selected_features":selected_features})
    pipeline_kwargs.update({"trial":trial})
    return full_pipeline.build_pipeline(params=params, model=model_name, cache=cache, **pipeline_kwargs)


def get_optimization_function(X_train, y_train, para_space_dict, best_features,
                              model_name="pytorch",
                              scoring="accuracy",
                              n_splits=3,
                              n_jobs_cv=1,
                              feature_handling_para=None,
                              pipeline_kwargs={}):
    """
    Build an Optuna-compatible objective function that creates, fits, and scores a pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix used for training and cross-validation.
    y_train : pd.Series or np.ndarray
        Target labels for classification.
    para_space_dict : dict
        Dictionary defining the Optuna parameter search space.
        Format: {"param_name": ("type", val1, val2)}, where type is one of
        {"uniform", "loguniform", "int", "categorical", "fixed", "float"}.
    best_features : list
        Ranked list of feature names (e.g., from SHAP analysis).
    model_name : str, optional
        Model identifier used by `get_model_pipeline()`. Default is "pytorch".
    scoring : str, optional
        Scoring metric used for `cross_val_score`. Default is "accuracy". See https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names for available scores.
    n_splits : int, optional
        Number of cross-validation folds. Default is 3.
    n_jobs_cv : int, optional
        Number of jobs to use in `cross_val_score`. If >1, caching is disabled. Default is 1.
    feature_handling_para : dict, optional
        Strategy for feature space exporation. Accepted formats:
        - {"type": "top_k_fixed", "top_k_min_fixed": int}
        - {"type": "top_k_explore", "top_k_min_fixed": int, "top_k_limit": [low, high], "exp_decay": float}
    pipeline_kwargs : dict, optional
        Additional kwargs for the pipeline. The definition of the feature type groupings (target encoding, text features, etc) must be define here.
        See the full_pipeline.get_model_pipeline function for more info.

    Returns
    -------
    callable
        Objective function compatible with Optuna’s `study.optimize()`.
    """

    if not pipeline_kwargs:
        logger.warning("No feature type given as input")
    
    # check parameters agains feature list
    pipeline_helpers.check_para(para_space_dict, pipeline_kwargs)

    if feature_handling_para is None:
        feature_handling_para = {"type": "top_k_fixed", "top_k_min_fixed": 50}

    # use pipeline caching if not multitread
    if n_jobs_cv == 1:
        pipeline_cache = True
    

    # create info string and check feature_handling_para inputs
    # -----------------------------------------------------------------------------------------------------------------------------------------

    log_info_str = "Creating an objective function"

    # explore feature space
    if feature_handling_para["type"] == "top_k_explore":
        if (feature_handling_para["top_k_min_fixed"] is None) \
            or (feature_handling_para["top_k_min_fixed"] > feature_handling_para["top_k_limit"][0]) \
            or (feature_handling_para["top_k_min_fixed"] > len(best_features)) \
            or (feature_handling_para["top_k_limit"][1] > len(best_features)):
            raise ValueError("Wrong values for top_k_min_fixed or top_k_limit for feature exploration: \
                            top_k_min_fixed and top_k_limit must be non None and smaller than length of the best_features list")
        else:
            log_info_str += f' which will explore the feature space between rank {str(feature_handling_para["top_k_limit"][0])} \
                and rank {str(feature_handling_para["top_k_limit"][1])} while always keeping the first {str(feature_handling_para["top_k_min_fixed"])} features'

    # take all features given as input in pipeline_kwargs
    elif feature_handling_para.get("top_k_min_fixed", None) is None:
        log_info_str += " with all features from the best_features list given as input in pipeline_kwargs."

    # take the first top_k_min_fixed features given as input
    elif isinstance(feature_handling_para["top_k_min_fixed"], int):
        log_info_str += f" with the first {feature_handling_para['top_k_min_fixed']} features from the best_features list given as input."
    else:
        raise ValueError(f'top_k_min_fixed must be either an integers or None, got {feature_handling_para["top_k_min_fixed"]}')
    logger.info(log_info_str)


    # define optuna's objective function dynamically
    # -----------------------------------------------------------------------------------------------------------------------------------------

    def objective(trial):

        # Feature space selection
        if feature_handling_para["type"] == "top_k_explore":
            top_k = trial.suggest_int("top_k", *feature_handling_para["top_k_limit"])
            selected_features = tuning_helpers.sample_feature_subset(
                trial, top_k,
                top_k_min_fixed=feature_handling_para["top_k_min_fixed"],
                best_features=best_features,
                decay=feature_handling_para["exp_decay"]
            )
            trial.set_user_attr("selected_features", selected_features)
            trial.set_user_attr("selected_feature_indices", [best_features.index(f) for f in selected_features])
        else:
            k = feature_handling_para.get("top_k_min_fixed", len(best_features))
            selected_features = best_features[:k]

        # Parameter space
        params = {}
        for name, spec in para_space_dict.items():
            kind = spec[0]
            if kind == "uniform":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == "loguniform":
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
            elif kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
            elif kind == "fixed":
                params[name] = spec[1]
                trial.set_user_attr(name, spec[1]) # store fixed parameters as usr attribute
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            else:
                raise ValueError(f"Unknown sampling type: {kind}. Must be either 'uniform', 'loguniform', 'int', 'categorical', 'float' or 'fixed'")

        # Construct pipeline with sklearn CV multi threads
        if n_jobs_cv > 1:
            # creat new pipeline instance to avoid issue between caching and parralelization
            model_pipeline = _build_pipeline(params, model_name, selected_features, trial=trial, 
                                            cache=False, pipeline_kwargs=pipeline_kwargs)

        # Cache pipeline if not sklearn CV multi thread 
        else:
            model_pipeline = full_pipeline.get_model_pipeline(model=model_name, selected_features=selected_features, 
                                            cache=True, trial=trial, **pipeline_kwargs)
            params_filtered = pipeline_helpers.filter_valid_params(model_pipeline, params)
            model_pipeline.set_params(**params_filtered)

        # perform cross validation with prunning
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
            scores = cross_val_score(model_pipeline, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs_cv)
        except ValueError as e:
            # Check if TrialPruned is buried inside Error
            if "TrialPruned" in str(e):
                raise optuna.exceptions.TrialPruned("Trial was pruned inside a CV fold.")
            # other error
            else:
                logger.error("Trial failed due to: ", repr(e))
                return float("nan")

        # return mean of the cv scores
        return np.mean(scores)

    return objective


class OptRes:
    """
    Wrapper class for an Optuna Study object. Provides utilities for accessing
    best parameters, plotting parameter importance, and visualizing search results.
    """

    def __init__(self, study):
        """
        Parameters
        ----------
        study : optuna.Study
            Completed Optuna study object.
        """
        self.study = study
        self.best_params = study.best_trial.params

    def get_params_plot(self, other_para_to_show=None):
        """
        Generate an interactive parallel coordinate plot (if supported).

        Parameters
        ----------
        other_para_to_show : dict, optional
            Additional metrics to include in the visualization (for comparing multiple runs for example)

        Returns
        -------
        plotly.graph_objs.Figure
            Plotly figure object for parameter visualization.
        """
        return tuning_helpers.plot_parallel(self.study, res_dict=other_para_to_show)[0]

    def get_params_importance_plot(self):
        """
        Generate a bar plot of parameter importances.

        Returns
        -------
        matplotlib.figure.Figure
            Bar chart showing relative importance of each parameter.
        """
        para_impor = get_param_importances(self.study)
        fig = plt.figure()
        plt.barh(list(para_impor.keys())[::-1], list(para_impor.values())[::-1])
        plt.ylabel("Hyperparameters")
        plt.xlabel("Relative importance")
        plt.close()
        return fig

    def get_best_pipeline_params(self):
        """
        Return best trial parameters and metadata (e.g., selected features).

        Returns
        -------
        dict
            Combined dictionary of best parameters and user attributes.
        """
        best_trial = self.study.best_trial
        out = best_trial.params.copy()
        out.update(best_trial.user_attrs)
        return out


def optuna_optimize(objective_function, n_trials=50, study_name=None, log_file_name=None, direction="maximize", 
                    n_jobs_optuna=1, sampler="TPES", pruning=True):
    """
    Run Optuna optimization with support for logging, checkpointing, and parallel trials.

    Parameters
    ----------
    objective_function : callable
        Optuna-compatible objective function created by `get_optimization_function`.
    n_trials : int, optional
        Number of optimization trials. Default is 50.
    study_name : str or None, optional
        Name of study. If provided, results are saved in `optuna_studies/{study_name}.db`.
    log_file_name : str or None, optional
        File name for logging output. If None, no file logging.
    direction : str, optional
        Optimization direction: "maximize" or "minimize". Default is "maximize".
    n_jobs_optuna : int, optional
        Number of Optuna trials to run in parallel. Default is 1.
    sampler : str, optional
        Sampler strategy for Optuna. Options include {"TPES", "Grid", "Random", "CMA", etc.}.
    pipeline_verbose : bool, optional
        If False, disables pipeline logging. Default is False.
    pruning: bool, optional
        If True uses Median Pruner, if False NoPruner

    Returns
    -------
    OptRes
        A wrapped object containing the final Optuna study and utility methods.
    """


    # Checks
    # ------------------------------------------------------------------------------------------------------------------------

    # check if optuna_studies directory exist
    if (not os.path.isdir('optuna_studies')) and (study_name is not None):
        os.makedirs('optuna_studies')
        logger.info(f"Created 'optuna_studies' directory to store optuna's results in {study_name}.")

    # check optuna sampler algorithm choice
    if sampler not in optuna_algorithm_dict.keys():
        err_str = f"Wrong sampler type for optuna. Available sampler methods:{list(optuna_algorithm_dict.keys())} , got {sampler}"
        logger.error(err_str)
        raise ValueError(err_str)
    else:
        sampler = optuna_algorithm_dict[sampler]
        logger.info(f"Hyper parameter tuning using Optuna's {sampler} sampler")
    

    # define pruning approach
    # ------------------------------------------------------------------------------------------------------------------------

    if pruning:
        pruner = optuna.pruners.MedianPruner()
    else:
        pruner = optuna.pruners.NopPruner()


    # loggings
    # ------------------------------------------------------------------------------------------------------------------------

    # catch optuna logs
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.DEBUG)
    optuna_logger.propagate = False  # Important to avoid printing to stdout via root logger

    file_sink_id = None # pipeline sink
    optuna_file_handler = None

    if log_file_name is not None:
        logger.info(f"Addind sink files {log_file_name}_pipeline.log and {log_file_name}_optuna.log to log/ for pipeline and optuna logs, respectively.")
        # add pipeline logger sink to file and remove stdout
        file_sink_id = logger.add(f"logs/{log_file_name}_pipeline.log", backtrace=True, diagnose=True, enqueue=True, level="DEBUG")
        # add optuna's logger sink to file
        optuna_file_handler = logging.FileHandler(f"logs/{log_file_name}_optuna.log", mode='a')  # or 'w' to overwrite
        optuna_logger.addHandler(optuna_file_handler)
    
    else:
        # optuna to stdout
        optuna_logger.addHandler(logging.StreamHandler(sys.stdout))

    # Optuna trial storage
    # ------------------------------------------------------------------------------------------------------------------------

    storage = None
    if study_name is not None:
        path = f"optuna_studies/{study_name}.db"
        storage = f"sqlite:///{path}"


    # create and run study
    # ------------------------------------------------------------------------------------------------------------------------

    study = optuna.create_study(study_name=study_name, storage=storage, direction=direction, load_if_exists=True, 
                                sampler=sampler(), pruner=pruner)

    study.optimize(objective_function, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs_optuna)


    # closing logging handlers
    # ------------------------------------------------------------------------------------------------------------------------

    if file_sink_id is not None:
        logger.remove(file_sink_id)

    if optuna_file_handler is not None:
        optuna_logger.removeHandler(optuna_file_handler)
        optuna_file_handler.close()
    
    return OptRes(study)




if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import random
    import pandas as pd

    # Fake data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_df = pd.DataFrame(X)
    # add random categorical feature
    X_df["categorical"] = np.random.choice(["a", "b", "c", "d"], len(X))
    X_df.columns = X_df.columns.astype(str)

    X_train, _, y_train, _ = train_test_split(X_df, y, test_size=0.3)

    # Dummy feature names
    best_features = list(X_df.columns[:5])
    best_features.append("categorical")

    # feature type definition
    pipeline_kwargs = {
        "cat_to_targetencode_A": ["categorical"],
        "num_to_logtransform": ["feat_0", "feat_2"]
    }

    # Dummy param space
    search_space = {
        "classifier__base_estimator__lr": ("loguniform", 1e-5, 1e-3),
        "classifier__base_estimator__dropout": ("uniform", 0.1, 0.5),
        "classifier__base_estimator__batch_size": ("int", 64, 512),
        "classifier__base_estimator__hidden_activation": ("categorical", ["relu", "tanh"]),
        "categorical_encoding__te_A__smoothing": ("fixed", 5)
    }

    # Objective without feature space exporation
    objective_fn = get_optimization_function(
        X_train, y_train,
        para_space_dict=search_space,
        best_features=best_features,
        model_name="pytorch",
        scoring="accuracy",
        feature_handling_para={
            "type": "top_k_fixed"
        },
        pipeline_kwargs=pipeline_kwargs
    )

    results = optuna_optimize(objective_fn, n_trials=3, pipeline_verbose=False)
    print()
    print("------------------------------------------------------------")
    print("Optuna optimization without feature space exploration sucess")
    print("------------------------------------------------------------")
    print("Best parameters:", results.get_best_pipeline_params())
    print()

    objective_fn = get_optimization_function(
        X_train, y_train,
        para_space_dict=search_space,
        best_features=best_features,
        model_name="pytorch",
        scoring="accuracy",
        feature_handling_para={
            "type": "top_k_explore",
            "top_k_min_fixed": 3,
            "top_k_limit": [4, 5],
            "exp_decay": 0.05
        },
        pipeline_kwargs=pipeline_kwargs
    )

    results = optuna_optimize(objective_fn, n_trials=3, pipeline_verbose=False)
    print()
    print("------------------------------------------------------------")
    print("Optuna optimization with feature space exploration sucess")
    print("------------------------------------------------------------")
    print("Best parameters:", results.get_best_pipeline_params())
    print()

