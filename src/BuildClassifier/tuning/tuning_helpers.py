import paxplot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import numpy as np
from copy import deepcopy

import random
from itertools import product



def plot_parallel(study, res_dict=None):
    """
    Plot parrallel chart using all data from a optuna.study

    Args:
        study_df (pd.DataFrame): dataframe with the parameter and the score (value) as column and the different trials as rows.
        res_dict (dict, optional): dictionary of parameters with parameter names as key and value as value. Only the parameters present in study.params will be displayed. 
                                    Defaults to None.
    """

    # store history of the optuna study in df
    study_df = pd.DataFrame(data=[std.params for std in study.trials if std.values is not None])

    # best parameters (post optuna) as dataframe
    best_df = study.best_params
    best_df = pd.DataFrame(data=[best_df])

    # check format of the NN architecture if present
    if "min_layer_units_0" in study_df.columns:
        nn_layer_cols = [c for c in study_df.columns if "units" in c]
        nn_layer_bool_to_drop = [c for c in study_df.columns if "active" in c]
        study_df["classifier__base_estimator__dense"] = study_df[nn_layer_cols].apply(lambda row: "_".join([str(round(i)) for i in row if i == i]), axis=1)
        best_df["classifier__base_estimator__dense"] = study_df[nn_layer_cols].apply(lambda row: "_".join([str(round(i)) for i in row if i == i]), axis=1)
        study_df.drop(columns=[c for c in nn_layer_cols+nn_layer_bool_to_drop if c in study_df.columns], inplace=True)
        best_df.drop(columns=[c for c in nn_layer_cols+nn_layer_bool_to_drop if c in best_df.columns], inplace=True)
    
    # add score
    study_df["values"] = pd.DataFrame(data=[std.values for std in study.trials if std.values is not None]).values
    best_df["values"] = study.best_value
    
    cols = study_df.columns

    # deal with dense layer and dropout format from pipeline 
    if res_dict is not None:
        res_dict_cp = deepcopy(res_dict)
        for res_name, para_dict in res_dict_cp.items():
            if "classifier__base_estimator__dense" in para_dict.keys():
                if isinstance(res_dict_cp[res_name]["classifier__base_estimator__dense"], list):
                    res_dict_cp[res_name]["classifier__base_estimator__dense"] = "_".join([str(i) for i in res_dict_cp[res_name]["classifier__base_estimator__dense"]])
            if "classifier__base_estimator__dropout" in para_dict.keys():
                if isinstance(res_dict_cp[res_name]["classifier__base_estimator__dropout"], list):
                    res_dict_cp[res_name]["classifier__base_estimator__dropout"] = res_dict_cp[res_name]["classifier__base_estimator__dropout"][0]
            if "classifier__base_estimator__hidden_activation" in para_dict.keys():
                if isinstance(res_dict_cp[res_name]["classifier__base_estimator__hidden_activation"], list):
                    res_dict_cp[res_name]["classifier__base_estimator__hidden_activation"] = res_dict_cp[res_name]["classifier__base_estimator__hidden_activation"][0]
    else:
        res_dict_cp = None

    # Convert categorical columns to numeric indices and store mappings
    category_maps = {}  # maps column -> {label -> index}
    # loop over parameters given in study
    for col in study_df.columns:
        # if parameter value is string
        if study_df[col].dtype == object:
            unique_vals = sorted(study_df[col].dropna().unique()) # unique cqtegories
            cat_map = {v: i/(len(unique_vals)-1) for i, v in enumerate(unique_vals)} # create normalized ordinal encoding
            # map for both trials and best parameters
            study_df[col] = study_df[col].map(cat_map)
            best_df[col] = best_df[col].map(cat_map)
            # map all given parameter dictionaries
            if res_dict_cp is not None:
                # loop over result dict
                for res_name, para_dict in res_dict_cp.items():
                    # loop over parameters
                    for para_name, para_val in para_dict.items():
                        if col in para_name:
                            res_dict_cp[res_name][para_name] = cat_map[res_dict_cp[res_name][para_name]]
            category_maps[col] = cat_map
    
    # Create figure
    paxfig = paxplot.pax_parallel(n_axes=len(cols))
    paxfig.plot(study_df.to_numpy(), line_kwargs={"alpha":0.2})

    # Add colorbar for optuna trials
    color_col = len(cols)-1
    paxfig.add_colorbar(
        ax_idx=color_col,
        cmap='viridis',
        colorbar_kwargs={'label': cols[color_col]}
    )

    # Add line for optuna best param 
    paxfig.plot(
    best_df.to_numpy(),
    line_kwargs={'alpha': 1, 'color': "black", 'zorder': 10, 'linewidth':1, 'marker':'o', 'markersize':10, 'label':"optuna"}
    )

    # add line for other parameters given in para_dict
    if res_dict_cp is not None:
        color_list=["red", "blue", "green", "orange"]
        i = 0
        # loop over the para_dict to show other results
        for res_name, para_dict in res_dict_cp.items():
            para = para_dict#[p.replace("base_estimator", "") for p in para_dict.keys()] # store parameters and remove base_estimator to make it compatible with optuna output
            missing_para = [k for k in cols if k not in para] # check for parameters that appears in study but not in given res_dict key
            if missing_para:
                raise ValueError(f"Parameters {missing_para} from {res_name} are missing.")
            # take only relevant parameters from value -> dict (and deal with dorpout list format)
            value_list = [para_dict[k] for k in cols]
            value_formated = pd.DataFrame(data=[dict(zip(cols,value_list))])
            # replace missing parameter by 0
            if any(value_formated.isnull().values[0]):
                print("Warning, NAN value detected in the parameters list for {}, will be replaced by 0".format(res_name))
            value_formated.fillna(0, inplace=True)
            # plot parrallel plot
            paxfig.plot(
                value_formated.to_numpy(),
                line_kwargs={'alpha': 1, 'color': color_list[i], 'zorder': 10, 'linewidth':1, 'marker':'o', 'markersize':10, 'label':res_name}
            )
            i += 1

    # Add labels
    paxfig.set_labels(cols)

    # Add custom yticks for categorical axes
    def make_formatter(inv_map):
        def formatter(val, pos):
            return inv_map.get(val, "")
        return formatter

    for i, col in enumerate(cols):
        if col in category_maps:
            inv_map = {v: k for k, v in category_maps[col].items()}
            paxfig.axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(make_formatter(inv_map)))


    # rotate xticks
    for ax in paxfig.axes:
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')

    paxfig.axes[-2].legend() # plot legend of last axis
    paxfig.tight_layout()
    paxfig.figure.set_figheight(5)
    paxfig.figure.set_figwidth(15)
    #paxfig.legend()

    #plt.show()
    plt.close()

    return paxfig, best_df

def cross_valid(X_val, y_val, n_splits, score_func):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    scores = []
    for train_idx, val_idx in cv.split(X_test, y_test):
        X_train_tmp, X_val_tmp = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_tmp, y_val_tmp = y_train.iloc[train_idx], y_train.iloc[val_idx]

        mp = build_pipeline(para)
        mp.fit(X_train_tmp, y_train_tmp)

        y_pred = mp.predict_proba(X_val_tmp)[:, 1]
        score = score_func(y_val_tmp, y_pred)
        scores.append(score)
    
    return scores

def sample_feature_subset(trial, top_k, top_k_min_fixed, best_features, decay=0.1):
    """Sample features: top_k_moin_fixed + sample from exponential tail."""

    top_k_fixed_features = best_features[:top_k_min_fixed]
    remaining_features = best_features[top_k_min_fixed:]
    
    rng = random.Random(trial.number)  # ensures deterministic per-trial sampling

    n_sampled = top_k - top_k_min_fixed
    ranks = list(range(len(remaining_features)))
    weights = np.exp(-decay * np.array(ranks))
    weights /= weights.sum()

    sampled = np.random.choice(remaining_features,n_sampled,replace=False, p=weights).tolist()    
    
    return top_k_fixed_features + sampled

def build_architecture(trial, min_layers, max_layers):
    """
    Use bool and integers variable to dynamically build a NN architecture through optuna.
    """
    dense = []
    for i in range(max_layers):
        use_layer = trial.suggest_categorical(f"layer_{i}_active", [True, False])
        if use_layer:
            neurons = trial.suggest_int(f"layer_{i}_units", 64, 512, step=32)
            dense.append(neurons)
    # Enforce minimum number of layers
    if len(dense) < min_layers:
        for _ in range(min_layers - len(dense)):
            neurons = trial.suggest_int(f"min_layer_units_{_}", 32, 256, step=32)
            dense.append(neurons)
    return dense

def generate_architectures(min_layers, max_layers, min_neurons, max_neurons, step=32):
    """
    Create a list of all possible NN architecture that satisfies the following criteria:
    - minimum number of layers 
    - maximum number of layers
    - minimum number of neurons per layers
    - maximum number of neurons per layers
    - number of neurons step of 32
    - Returns decreasing deep layer size
    """
    candidates = []
    possible_sizes = list(range(min_neurons, max_neurons + 1, step))

    for depth in range(min_layers, max_layers + 1):
        for combo in product(possible_sizes, repeat=depth):
            if all(combo[i] >= combo[i+1] for i in range(len(combo)-1)):  # decreasing or flat
                candidates.append("_".join(map(str, combo)))
    return candidates