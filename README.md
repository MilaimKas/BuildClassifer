# 📦 BuildClassifier: Modular Pipeline and Model Tuning Framework

`BuildClassifier` is a modular machine learning framework that integrates:

- 🔌 Flexible scikit-learn pipelines  
- ⚙️ Support for multiple model backends (PyTorch, XGBoost, Random Forest)  
- 🎯 Optuna-based hyperparameter tuning
- 🧠 Advanced preprocessing (free text embeddings, clipping, log transform, normalization) in a sklearn pipeline  
- 🧪 Support for early stopping and pruning  
- 📊 Feature importance tracking and output calibration (e.g., Expected Precision)
- ↔️ Uncertainty aware option (experimental)

---

## 📁 Package Structure

```text

|____pipelines
| |____logger_wrapper.py                # pipeline steps for logging and debugging
| |____pipeline_parameters.py           # pipeline parameters examples
| |____pipeline_helpers.py              # safe_column_transform, filering utilities
| |____full_pipeline.py                 # get_model_pipeline() and build_pipeline()
|____tuning
| |____optuna_tuning.py                 # wrapper function for simpler tuning
| |____metrics_for_optimization.py      # "non-conventional" metrics for tuning
| |____tuning_helpers.py                # utilities for feature sapce sampling and optuna result plotting
| |______init__.py
|____models
| |____model_testing.py                 # class for analysing classifier performance
| |____feature_importance.py            # wrapper for permutation, SHAP and model feature importance
| |____Pytorch_model_wrapper.py         # custom class for PyTorch classifier 
| |____Tree_model_wrapper.py         # custom class for XGBoost classifier
| |____calibration_wrapper.py           # wrapper to callibrated classifier (expected precision)
|____feature_selection
| |____feature_selector.py              # feature selection class
| |____feature_selection_utilities.py   # utilities for feature analysis such as correlation
| |______init__.py
|____feature_engineering
| |____feature_creation_utilities.py            # helper function to transform features
| |____missing_column_creator.py                # transformer to create a missing count column
| |____feature_transformation_utilities.py      # class for feature transformations (ClipTransformer)
| |____decorators.py                            # function decorators
| |____text_classifier_model.py                 # Free-text embedding submodel
| |______init__.py

```

## 🚀 Quick Start

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Create and train a model

In order to train a model, if needed, a FeatureCreation class that transforms the raw data into usable feature, must be provided. 
If the pre-processing step is to be incorporated into the pipeline (for example, at inference step), the class should be a sklearn compatible transformer
that contains a fit and transform method. 

Minimal example:
```python

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from BuildClassifier.feature_engineering.feature_creation_utilities import onehot_encode_with_vocab

from config import cat_to_onehot

from loguru import logger

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Usage as pre-processing:
    X_transformed = FeatureCreator(X_raw, y_raw).transform()
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # create time related features
        X_new = pd.concat([self.create_time_features(X), X_new], axis=1)
        logger.success("Created time-related features")

        # onehot encoding
        X_new = pd.concat([self.create_one_hot_encoding_features(X_new, cat_to_onehot=cat_to_onehot), X_new], axis=1)
        X_new.drop(columns=list(cat_to_onehot.keys()), inplace=True)
        logger.success(f"Created one hot encoding for {list(cat_to_onehot.keys())}")

        logger.success("FeatureCreator transformation done")

        return X_new
    
    def create_time_features(self, X):
        
        X_time = pd.DataFrame(index=X.index)
        X_time["time"] = feat_creation.create_day_time(X["raw_timestamp"])

        return X_time

    def create_one_hot_encoding_features(self, X, cat_to_onehot=cat_to_onehot):
        return onehot_encode_with_vocab(X, cat_to_onehot)


```

Training and analysing the model:

```python

from sklearn.model_selection import train_test_split
from BuildClassifier.pipelines import full_pipeline
from BuildClassifier.models.model_testing import ModelTesting
from BuildClassifier.tuning import optuna_tuning
from BuildClassifier.models import feature_importance

# create all features for the entire dataset (!!! The FeatureCreator class must be provided !!!)
X_transformed = FeatureCreator().transform(X)

# splt test train
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

# Train RandomForest model

# define parameters
params = {
    # xgbosst parameters
    "classifier__base_estimator__learning_rate": 0.05,
    "classifier__base_estimator__max_depth": 20,
    "classifier__base_estimator__subsample": 0.8,
    "classifier__base_estimator__colsample_bytree":0.8,
    "classifier__base_estimator__reg_alpha": 0,
    "classifier__base_estimator__reg_lambda": 1,
    "classifier__base_estimator__gamma": 0,
    "classifier__base_estimator__min_child_weight": 1,
    "classifier__base_estimator__n_estimators": 200,
    # pipeline transformation parameters
    'Free_text_embedding_model__rf_howdidyou_embeddings__n_estimators': 30,
    'Free_text_embedding_model__rf_howdidyou_embeddings__max_depth': 15,
    'Free_text_embedding_model__rf_occupation_embeddings__n_estimators': 150,
    'Free_text_embedding_model__rf_occupation_embeddings__max_depth': 15,
    'categorical_encoding__te_A__smoothing': 4.104737351688896,
    'categorical_encoding__te_B__smoothing': 10.011479543015225,
    'categorical_encoding__te_C__smoothing': 25.10511396415008}

# add selected features to the parameter dictionary
params.update({"selected_features":best_features[:50]})

# create pipeline
model_pipeline = full_pipeline.build_pipeline(params, model_name="randomforest", pipeline_kwargs=feat_group_dict)

# train pipeline
model_pipeline.fit(X_train, y_train)

# get predictions
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:,1]
y_expect_pre = model_pipeline.predict_expected_precision(X_test)

```
### 3. Model performance
```python

logger.disable("")

# create Modeltesting object for two test sets
testing = ModelTesting(y_test, X_test, model_pipeline)
testing_val = ModelTesting(y_val, X_val, model_pipeline)

# get standard classification scores
res = testing.get_scores(use_expected_prec=False, cut=0.70)
res_val = testing_val.get_scores(use_expected_prec=False, cut=0.70)
display(res)
display(res_val)

# PR and ROC curves
display(testing.pr_curve())
display(testing.roc_curve())

# calibration curves
_ = testing.inspect_callibration()

# get score uncertainity using bootstrapping
boot_res = testing.score_uncertainity(cut=0.7, n_samples=50)
# check std for all scores
boot_res.std(axis=0)*100*2
# visualize distribution
plt.hist(boot_res["fp_share"]*100)
plt.xlabel("False positive rate (%)")
plt.show()

```

### 4. Hyperparameter tuning

```python

# define space parameters
search_space = {'classifier__base_estimator__n_estimators': ("int", 100, 300),
 'classifier__base_estimator__max_depth': ("int", 10, 25),
 'Free_text_embedding_model__rf_howdidyou_embeddings__n_estimators': ("int", 30,100),
 'Free_text_embedding_model__rf_howdidyou_embeddings__max_depth': ("int", 10, 25),
 'Free_text_embedding_model__rf_occupation_embeddings__n_estimators': ("int", 50, 150),
 'Free_text_embedding_model__rf_occupation_embeddings__max_depth': ("int", 10, 25),
 'categorical_encoding__te_A__smoothing': ("fixed", 4.),
 'categorical_encoding__te_B__smoothing': ("fixed", 10),
 'categorical_encoding__te_C__smoothing': ("fixed", 25)
}

# Objective without feature space exporation
objective_fn = optuna_tuning.get_optimization_function(
    X_train, y_train,
    para_space_dict=search_space,
    best_features=best_features,  # all features
    model_name="randomforest",
    scoring="accuracy",
    feature_handling_para={
        "type": "top_k_fixed"
    },
    pipeline_kwargs=feat_group_dict
)

# run optimization
optuna_results = optuna_tuning.optuna_optimize(objective_fn, n_trials=50, pipeline_verbose=False, n_jobs_optuna=8,
                        study_name="rf_full_features_tefixed", log_file_name="rf_tuning")

# check results
optuna_results.get_params_plot()
optuna_results.get_params_importance_plot()
optuna_results.get_best_pipeline_params()

```

### 5. Feature importance

```python
featimp = feature_importance.FeatureImportance(model_pipeline, X_val)

# get importances
_ = featimp.get_permutation_importances(y=y_val)
_ = featimp.get_shap_values()
_ = featimp.get_model_importances()

# display plot and dataframe with normalized importance
fig = featimp.plot_importance(permutation=True)
display(fig)

display(featimp.feature_importance_df)

```

### 6. uncertainty-aware model (experimental)

We have implemented uncertainty prediction for the PyTorch and RandonForest model, when the classifier argument 'use_beta' is set to True.
This makes the model uncertainty-aware: it outputs a Beta distribution instead of a point prediction for each data points. This means that, in addition to the raw probability output, we also have acces to it's uncertainty (through the variance of the Beta distribution).

Minimal example:

```python

# make model uncertainty-aware
params.update({"classifier__base_estimator__use_beta":True})

pipeline = build_pipeline(params, "randomforest", cache=False, pipeline_kwargs=pipeline_kwargs)
print("Pipeline with uncertainity-aware randomforest model created successfully.")

pipeline.fit(X_train, y_train)
print("Pipeline succesfully trained")

# transform for inference
X_val_transformed = pipeline[:-1].transform(X_val)

# get uncertainty-aware output (DataFrame with mean (mu) and variance (var))
uncert_df = pipeline.named_steps["classifier"].predict_uncert(X_val_transformed)
print(uncert_df.head(10))

```
