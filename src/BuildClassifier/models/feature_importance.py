import shap
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch.nn as nn
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.inspection import permutation_importance


class FeatureImportance:
    """
    Wrapper class for calculating and visualizing feature importance using:
    - SHAP values (TreeExplainer, DeepExplainer)
    - Model-internal feature importance (e.g., from RandomForest)
    - Permutation importance (sklearn)

    Experimental: std output from uncertainty-aware models

    Supports RandomForest, XGBoost, and PyTorch models.
    """

    def __init__(self, model_pipeline, X=None, X_transformed=None, background_samples_max=1000, shap_uncert=False):
        """
        Parameters
        ----------
        model_pipeline : sklearn.Pipeline
            Trained pipeline including a feature selection/transformation step and a final classifier wrapped in a calibration class.
        X : pd.DataFrame
            Input features.
        background_samples_max : int, optional
            Maximum number of samples to use for SHAP background (DeepExplainer and Explainer), by default 1000.
        """

        background_samples = min(background_samples_max, len(X))

        self.model_pipeline = model_pipeline

        # get transfomed features
        if X_transformed is None:
            if X is None:
                raise ValueError("Need train data as input, either raw data or already transformed data.")
            self.X = model_pipeline[:-1].transform(X)
        else:
            self.X = X_transformed

        self.feature_names = model_pipeline["feature_selection"].cols
        self.base_estimator = model_pipeline.named_steps["classifier"].base_estimator

        self._initialize_explainer(shap_uncert, background_samples)

        self.shap_samples = None
        self.shap_values = None
        self.shap_values_uncert = None
        self.model_importances = None
        self.permutation_importance = None
        self.permutation_importance_uncert = None
        self.feature_importance_df = None


    def _initialize_explainer(self, shap_uncert, background_samples):
        """
        Model detection and explainer setup
        """
        
        background = self.X.sample(background_samples, replace=False).to_numpy()

        # RandomForest model
        # ----------------------------------------------------------------------

        if isinstance(self.base_estimator, RandomForestClassifier):

            logger.info("Base estimator is a RandomForest model")
            self.model = self.base_estimator
            self.model_type = "randomforest"
            self.explainer = shap.TreeExplainer(self.model)

            # add explainer for variance output
            if shap_uncert:
                logger.info(f"Adding uncertainty explainer for {self.model_type}. The current implementation used the model agnostic Kernel explainer. \
                                This may lead to an increase in computation time")
                if not self.model.use_beta:
                    err_str = "Attempting to get shap variance from a RandomForest model that is not uncertaity-aware. \
                                Define model with use_beta=True and re-train."
                    logger.error(err_str)
                    raise ValueError(err_str)

                # wrapper function to overwrire default use of model.predict for model output
                def predict_std(X):
                    X_df = pd.DataFrame(X)
                    return np.sqrt(self.model.predict_uncert(X_df)["var"].to_numpy())
                self.uncert_explainer = shap.Explainer(predict_std, background)

        elif hasattr(self.base_estimator, "model"):

            # XGBoost model
        # ----------------------------------------------------------------------

            if isinstance(self.base_estimator.model, XGBClassifier):

                self.model = self.base_estimator.model
                self.model_type = "xgboost"
                logger.info(f"Base estimator is a xgboost model")
                self.explainer = shap.TreeExplainer(self.model)

                if shap_uncert:
                    err_str = "XGboost model is not uncertainty-aware"
                    logger.error(err_str)
                    raise NotImplementedError(err_str)
            
            # PyTorch model
        # ----------------------------------------------------------------------

            elif self.base_estimator.model.__class__.__name__=="FullyConnectedNN":

                logger.info("Base estimator is a PyTorch model")

                self.model = self.base_estimator.model # PyTorch Fully connected object
                self.model_type = "pytorch"
                background_tensor = torch.tensor(background, dtype=torch.float32)

                # uncertainty-aware PyTorch model
                if shap_uncert:
                    logger.info("Adding uncertainty explainer for a uncertainty-aware PyTorch model")
                    # check if PyTorch is uncertainty aware
                    if not self.model.use_beta:
                        err_str = "Attempting to get shap variance from a PyTorch model that is not uncertainty-aware. Define model with use_beta=True and re-train."
                        logger.error(err_str)
                        raise ValueError(err_str)
                    
                    # define wrapper around pytorch to provide point prediction and std output from alpha/beta outputs
                    class SHAPWrapper(nn.Module):
                        def __init__(self, pytorch_classifier, std=True):
                            super().__init__()
                            self.beta_model = pytorch_classifier.model  # use the internal NN model
                            self.std = std
                        def forward(self, x):                          
                            alpha, beta = self.beta_model(x)
                            if self.std:
                                res = torch.sqrt((alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1)))
                            else:
                                res = alpha / (alpha + beta)
                            return res.unsqueeze(1)  # shape [batch_size, 1]

                    # std output
                    std_warp = SHAPWrapper(self.base_estimator)
                    self.uncert_explainer = shap.DeepExplainer(std_warp, background_tensor)
                    # mean output
                    mean_warp = SHAPWrapper(self.base_estimator, std=False)
                    self.explainer = shap.DeepExplainer(mean_warp, background_tensor)
                
                else:

                    # define really simple wrapper to unsqueezed model'output
                    class SHAPWrapper(nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, x):
                            out = self.model(x)
                            if out.ndim == 1:
                                return out.unsqueeze(1)  # shape: [batch_size, 1]
                            return out
                    mean_warp = SHAPWrapper(self.model)
                    self.explainer = shap.DeepExplainer(mean_warp, background_tensor)

            else: 
                raise ValueError(f"Base estimator not recognised: {self.base_estimator.model.__class__.__name__}")

        else:
            logger.warning(f"Unsupported base estimator: {self.base_estimator}. Will fall back to default model agnostic explainer. This may lead to an increase in computation time.")
            self.model = self.base_estimator
            self.model_type = "default"
            self.explainer = shap.Explainer(self.model)

            if shap_uncert:
                raise NotImplementedError

    def _get_shap_values(self, n_samples=1000, shap_uncert=False):
        """
        Compute SHAP values using the appropriate explainer.

        Parameters
        ----------
        n_samples : int
            Number of samples to use for explanation.

        Returns
        -------
        np.ndarray
            SHAP values of shape (n_samples, n_features).
        """

        n_samples = min(len(self.X), n_samples)
        X_sample = self.X.sample(n_samples, replace=False)
        self.shap_samples = X_sample

        # get point prediction explainer or variance explainer
        if shap_uncert:
            explainer = self.uncert_explainer
        else:
            explainer = self.explainer

        # get shap values
        try:
            if self.model_type == "pytorch":
                shap_values = explainer.shap_values(torch.tensor(X_sample.to_numpy(), dtype=torch.float32), check_additivity=False)
            else:
                shap_values = explainer.shap_values(X_sample, check_additivity=False)
        # should not happended with additivity check truned off
        except AssertionError as e:
            if "do not sum up to the model's output" in str(e):
                warn_str = "SHAP failed strict additivity check (likely due to nonlinear model output). Use permutation based values instead."
                logger.warning(warn_str)
                shap_values = None
            else: 
                logger.warning(str(e))
                shap_values = None

        # deal with different shap value shape
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # assume binary classification
        if len(shap_values.shape) == 3:
            if shap_values.shape[-1] > 1:
                shap_values = shap_values[:,:,1]
            else:
                shap_values = shap_values[:,:,0]

        return shap_values

    def get_shap_values(self, n_samples=1000):
        """ calls _get_shap_values for deterministic model """
        self.shap_values = self._get_shap_values(n_samples=n_samples, shap_uncert=False)
        return self.shap_values

    def get_model_importances(self):
        """
        Extract in-built model feature importances (e.g., from RandomForest or XGBoost).

        Returns
        -------
        np.ndarray
            Feature importance scores.
        """
        if hasattr(self.model, "feature_importances_"):
            self.model_importances = self.model.feature_importances_
        else:
            logger.warning(f"Model {self.model} does not support internal feature importances.")
            self.model_importances = None
        return self.model_importances

    def get_permutation_importances(self, y, n_repeats=5, n_jobs=1, max_samples=0.8):
        """
        Compute permutation importance using sklearn.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            True labels.
        n_repeats : int, optional
            Number of permutation repeats. Default is 5.
        n_jobs: int, optional
            Number of cpus used for parrellel computing
        max_samples: float, optional
            Ratio of samples to be used for permuation computation. Mus be between 0 and 1.

        Returns
        -------
        sklearn.inspection.PermutationImportance
            Fitted permutation importance object.
        """
        perimp = permutation_importance(
            self.base_estimator, self.X, y, n_repeats=n_repeats, n_jobs=n_jobs, max_samples=max_samples
        )
        self.permutation_importance = perimp.importances_mean

        return self.permutation_importance

    def get_shap_values_on_uncertainty(self, n_samples=1000):
        """ calls _get_shap_values function for uncertainit-aware models """
        self.shap_values_uncert = self._get_shap_values(n_samples=n_samples, shap_uncert=True)
        return self.shap_values_uncert

    def get_permutation_importances_on_uncertainty(self, y, n_repeats=5, n_jobs=1, max_samples=0.8):
        """
        Compute permutation importance using model output variance instead of point prediction.
 

        Parameters
        ----------
        y : np.ndarray or pd.Series
            True labels.
        n_repeats : int, optional
            Number of permutation repeats. Default is 5.
       n_jobs: int, optional
            Number of cpus used for parrellel computing
        max_samples: float, optional
            Ratio of samples to be used for permuation computation. Mus be between 0 and 1.

        Returns
        -------
        sklearn.inspection.PermutationImportance
            Fitted permutation importance object.
        """

        # wrapper to overwite default call to predict() method
        class UncertaintyScorer:
            def __init__(self, model):
                self.model = model
            def fit(self):
                return self
            def score(self, X, y=None):
                return -self.model.predict_uncert(X)["var"].mean()  # negate: lower uncertainty = better

        scorer = UncertaintyScorer(self.base_estimator)
        perimp = permutation_importance(
            scorer, self.X, y, n_repeats=n_repeats, n_jobs=n_jobs, max_samples=max_samples
        )
        self.permutation_importance_uncert = perimp.importances_mean

        return self.permutation_importance_uncert

    def plot_importance(self, n_samples=100, max_feat=25, figsize=(12, 10), permutation=False):
        """
        Plot normalized feature importances from SHAP, model, and optionally permutation importance.

        Parameters
        ----------
        n_samples : int
            Number of samples for SHAP calculation.
        max_feat : int
            Max number of features to show in plot.
        figsize : tuple
            Size of the output plot.
        permutation : bool
            Whether to include permutation importances.

        Returns
        -------
        matplotlib.axes.Axes
            Matplotlib bar plot of feature importances.
        """

        # Compute importances if missing
        if self.shap_values is None:
            logger.info("Calculating SHAP values")
            self.get_shap_values(n_samples=n_samples)
        if self.model_importances is None and self.model_type in ["xgboost", "randomforest"]:
            logger.info("Calculating model importances")
            self.get_model_importances()
        if permutation and self.permutation_importance is None:
            logger.warning("Permutation importance not available. Did you run get_permutation_importances()?")

        # get the right shape out of SHAP
        shap_importances = np.abs(self.shap_values).mean(axis=0).flatten()

        # Combine into DataFrame
        self.feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP importance': shap_importances,
        }).sort_values(by='SHAP importance', ascending=False)

        # minmax scaling
        def minmax(series):
            smin = series.min()
            smax = series.max()
            return series.apply(lambda x: (x - smin) / (smax-smin))

        self.feature_importance_df['SHAP importance'] = minmax(self.feature_importance_df['SHAP importance'])

        # add in-built model importance
        if self.model_importances is not None:
            self.feature_importance_df['Model importance'] = minmax(pd.Series(self.model_importances))
        
        # add permutation importance
        if permutation and self.permutation_importance is not None:
            self.feature_importance_df["permutation_importance"] = minmax(
                pd.Series(self.permutation_importance)
            )

        # Plot scaled feature importance
        plot = self.feature_importance_df[:max_feat].set_index('Feature').plot(kind='bar', figsize=figsize)
        plt.title("Scaled Feature Importance")
        plt.ylabel("Scaled Importances")
        plt.tight_layout()
        plt.show()

        return plot
    
    def plot_shap_summary(self):
        """ returns default shap summary plot """
        if self.shap_samples is None:
            raise ValueError("You must first call get_shap_values")
        return shap.summary_plot(self.shap_values, self.shap_samples)

    def plot_shap_dependance(self, feat_1, feat_2):
        """ plot shap values as a function of two features """
        if self.shap_samples is None:
            raise ValueError("You must first call get_shap_values")
        if (feat_1 not in self.X.columns) or (feat_2 not in self.X.columns):
            raise ValueError("Column not found in given X DataFrame")
        return shap.dependence_plot(feat_1, self.shap_values, self.shap_samples, interaction_index=feat_2)



if __name__ == "__main__":

    import pandas as pd
    import numpy as np
    import sys

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from loguru import logger

    from BuildClassifier.pipelines.full_pipeline import get_model_pipeline

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(sys.stderr, level="ERROR")
    logger.add(sys.stderr, level="WARNING")

    # Generate synthetic dataset
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    X["cat_feat"] = np.random.choice(["a", "b", "c"], size=len(X))
    X.columns = X.columns.astype(str)

    # Split for permutation importance
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Define feature groups
    pipeline_kwargs = {
        "cat_to_targetencode_A": ["cat_feat"],
        "num_to_logtransform": ["feat_0", "feat_2"],
    }

    selected_features = list(X.columns)

    # Build pipeline with RandomForest (change model="xgboost" or "pytorch")
    print()
    print("#######################################")
    print(" Explanation on point prediction")
    print("#######################################")
    print()

    pipeline = get_model_pipeline(
        model="randomforest",
        selected_features=selected_features,
        cache=False,
        **pipeline_kwargs
    )

    pipeline.fit(X_train, y_train)

    # Run Feature Importance
    featimp = FeatureImportance(model_pipeline=pipeline, X=X_val)

    print("Computing SHAP values for randomforest...")
    shap_val = featimp.get_shap_values()
    print(shap_val.shape)
    print()

    print("Computing model importances...")
    model_imp = featimp.get_model_importances()
    print(model_imp.shape)
    print()

    print("Computing permutation importances...")
    per_imp = featimp.get_permutation_importances(y_val)
    print(per_imp.shape)
    print()

    print("Plotting...")
    featimp.plot_importance(n_samples=100, max_feat=10, permutation=True)
    featimp.plot_shap_summary()
    featimp.plot_shap_dependance(feat_1=X_train.columns[2], feat_2=X_train.columns[3])
    print()


    print()
    print("#######################################")
    print(" PyTorch: Explanation on output variance")
    print("#######################################")
    print()

    model_para = {"dense":"5_5", 
            "hidden_activation":"relu", 
            "dropout":0.05, 
            "lr":0.01, 
            "epochs":5, 
            "batch_size":20,
            "use_beta":True}
    pipeline_kwargs["classifier_kwargs"] = model_para

    pipeline = get_model_pipeline(
        model="pytorch",
        selected_features=selected_features,
        cache=False,
        **pipeline_kwargs
    )

    # fit
    pipeline.fit(X_train, y_train)

    # create feature importance instance
    featimp = FeatureImportance(model_pipeline=pipeline, X=X_train, shap_uncert=True)

    print()
    print("Computing SHAP values for mean...")
    shap_mean = featimp.get_shap_values()
    print("shape = ", shap_mean.shape)
    print()

    print()
    print("Computing SHAP values for std...")
    shap_std = featimp.get_shap_values_on_uncertainty()
    print(shap_std.shape)
    print()

    