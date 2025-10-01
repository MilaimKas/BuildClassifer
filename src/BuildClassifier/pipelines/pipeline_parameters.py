
# pipeline transformation parameters
transformation_parameters = ['Free_text_embedding_model__rf_featname__n_estimators', 
        'Free_text_embedding_model__rf_featname__max_depth',
        'categorical_encoding__te_A__smoothing',
        'categorical_encoding__te_B__smoothing',
        'categorical_encoding__te_C__smoothing',
        '']

# pytorch parameters
pytorch_parameters = ['classifier__base_estimator__dense', 'classifier__base_estimator__hidden_activation',
                'classifier__base_estimator__epochs', 'classifier__base_estimator__dropout',
                'classifier__base_estimator__lr', 'classifier__base_estimator__batch_size']

# random forest parameters
randomforest_parameters = ['classifier__base_estimator__n_estimators',
                     'classifier__base_estimator__max_depth', 'classifier__base_estimator__min_samples_split',
                     'classifier__base_estimator__min_samples_leaf']

# xgboost parameters
xgboost_parameters = {"classifier__base_estimator__learning_rate",
    "classifier__base_estimator__max_depth",
    "classifier__base_estimator__subsample",
    "classifier__base_estimator__colsample_bytree",
    "classifier__base_estimator__reg_alpha",
    "classifier__base_estimator__reg_lambda",
    "classifier__base_estimator__gamma",
    "classifier__base_estimator__min_child_weight",
    "classifier__base_estimator__n_estimators"}

