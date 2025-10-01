"""
Module containing pyTorch estimator and wrappers
The strategy here was to make the nested estimator: ExpectedPrecisionClassifier(TorchClassifier(FullyConnectedNN))
sklearn, SHAP and Optuna compatible without relying on additional packages such as skorch.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from skorch.callbacks import Callback
from skorch import NeuralNetClassifier

from optuna.integration import SkorchPruningCallback
from optuna import exceptions


from loguru import logger


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, dense=[64, 32], hidden_activation=['relu', 'relu'], dropout=[0.1, 0.1]):
        super(FullyConnectedNN, self).__init__()
        layers = []
        prev_dim = input_dim # first layer dimension = feature dimension
        
        # loop over layers
        for dim, act, drop in zip(dense, hidden_activation, dropout):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self._get_activation(act))
            layers.append(nn.Dropout(drop))
            prev_dim = dim
        
        # add last layers
        # Note: Sigmoid activation for binary output -> taken care of when using BCEWithLogitsLoss loss criterion
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers) # build model

    def forward(self, x):
        return self.model(x).squeeze()

    def _get_activation(self, act):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }.get(act, nn.ReLU())  # Default to ReLU


class SkorchTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, dense=None, hidden_activation=None, dropout=None, lr=None, epochs=None, batch_size=None, trial=None):
        
        self.input_dim = input_dim
        self.dense = dense
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        self.trial = None
        self.classes_ = np.array([0, 1])

    def fit(self, X_in, y_in=None):

        X, y = self._convert_data(X_in, y_in)

        # check and define input layer dimension based on given feature dimension
        if self.input_dim is None:
            self.input_dim = X_in.shape[1]
        elif X.shape[1] != self.input_dim:
            err_str = "Skorch model input dimension mismatch: the number of features in fit differs from features in transform."
            logger.error(err_str)
            raise ValueError(err_str)

        self._check_inputs()

        callbacks = []
        if self.trial is not None:
            callbacks.append(SkorchPruningCallback(self.trial, monitor="val_acc"))

        self.net_ = NeuralNetClassifier(
            module=FullyConnectedNN,
            module__input_dim=self.input_dim or X.shape[1],
            module__dense=self.dense,
            module__hidden_activation=self.hidden_activation,
            module__dropout=self.dropout,
            criterion=nn.BCEWithLogitsLoss,
            max_epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            train_split=None,
            callbacks=callbacks,
            verbose=0
        )

        logger.success(f"Skorch model with input dimension {self.input_dim} built")

        logger.info("Fitting Skorch classifier: start ")
        self.net_.fit(X, y)
        logger.success("Fitting Skorch classifier: done ")

        # to make it work with sklearn cross_validtaion, the error must be catched in the optimization function
        #score = accuracy_score(y, self.predict(X))
        #if score < 0.8:
        #    raise exceptions.TrialPruned("accuracy too low.")

        return self

    def predict(self, X_in):
        X, _ = self._convert_data(X_in)
        return self.net_.predict(X)

    def predict_proba(self, X_in):
        X, _ = self._convert_data(X_in)
        return self.net_.predict_proba(X)

    def _convert_data(self, X, y=None):
        
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif not torch.is_tensor(X):
            raise ValueError(f"Feature data must be either a pandas DataFrame, a numpy array or torch tensors, got {X_in}") 

        if y is not None:
            y = y.to_numpy() if isinstance(y, pd.Series) else y
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = None
        return X, y

    def _check_inputs(self):

        # dense layers
        if self.dense is None:
            self.dense = [64, 32]
        elif isinstance(self.dense, str):
            self.dense = [int(i) for i in self.dense.split("_")]
        elif not isinstance(self.dense, (list, tuple)):
            logger.warning(f"dense must be a list, got {self.dense}. Fall back to default.")
            self.dense = [64, 32]
            
        # activation type
        if self.hidden_activation is None:
            self.hidden_activation = ["relu"]*len(self.dense)
        elif isinstance(self.hidden_activation, str):
            self.hidden_activation = [self.hidden_activation]*len(self.dense) 
        elif isinstance(self.hidden_activation, list): 
            if len(self.hidden_activation) != len(self.dense):
                logger.warning(f"Length of hidden_activation ({len(self.hidden_activation)}) does not match number of dense layer ({len(self.dense)}). Fall back to default.")
                self.hidden_activation = ["relu"]*len(self.dense)
        else:
            logger.error(f"hidden_activation must be a string or a list, got {self.hidden_activation} ")
            raise ValueError(f"hidden_activation must be a string or a list, got {self.hidden_activation} ")

        # dropout rate
        if self.dropout is None:
            self.dropout = [0.1]*len(self.dense)
        elif isinstance(self.dropout, float):
            self.dropout = [self.dropout]*len(self.dense)
        elif isinstance(self.dropout, list):
            if len(self.dropout) != len(self.dense):
                logger.warning(f"Length of dropout ({len(self.dropout)}) does not match length of dense ({len(self.dense)}). Fall back to default")
                self.dropout = [0.1]*len(self.dense)
        else:
            logger.error(f"dropout must be a float or a list, got {self.dropout}")
            raise ValueError(f"dropout must be a float or a list, got {self.dropout}")

        # leraning rate
        if self.lr is None:
            self.lr = 0.005
        elif not isinstance(self.lr, float):
            logger.error(f"Learning rate must be a flota, got {self.lr}")
            raise ValueError(f"Learning rate must be a flota, got {self.lr}")
        
        # epoch
        if self.epochs is None:
            self.epochs = 10
        elif not isinstance(self.epochs, int):
            logger.error(f"epochs must be an integers, got {self.epochs}")
            raise ValueError(f"epochs must be an integers, got {self.epochs}")

        # batch size
        if self.batch_size is None:
            self.batch_size = 128
        elif not isinstance(self.batch_size, int):
            logger.error(f"batch_size must be an integers, got {self.batch_size}")
            raise ValueError(f"batch_size must be an integers, got {self.batch_size}")



if __name__ == "__main__":

    # random data
    dummy_X_data = np.random.normal(1, 0.2, (1000, 10))
    dummy_y_data = np.random.binomial(1,0.2, 1000)

    para = {"input_dim":10, 
            "dense":"5_5", 
            "hidden_activation":"relu", 
            "dropout":0.05, 
            "lr":0.01, 
            "epochs":5, 
            "batch_size":20}
    skorch_model = SkorchTorchWrapper(**para)

    # fit
    skorch_model.fit(dummy_X_data, dummy_y_data)

    # predict
    skorch_predict = skorch_model.predict_proba(dummy_X_data)

    # plot
    print(skorch_predict)