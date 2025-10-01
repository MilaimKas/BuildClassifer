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

from optuna.integration import SkorchPruningCallback
from optuna import exceptions

from loguru import logger


class EarlyStopper:
    def __init__(self, patience=5, tolerance=0.001):
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        # Only reset if clearly better
        if validation_loss < self.min_validation_loss - self.tolerance:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, dense=[64, 32], hidden_activation=['relu', 'relu'], dropout=[0.1, 0.1], use_beta=False):
        """
        PyTorch NN model wrapper

        Args:
            input_dim (int): number of features (input dimension).
            dense (list, optional): architecture of the NN. Defaults to [64, 32].
            hidden_activation (list, optional): activation function for each layer. Defaults to ['relu', 'relu'].
            dropout (list, optional): dropout rate (float, 0-1) per layer. Defaults to [0.1, 0.1].
            use_beta (bool, optional): if True, train the model to fit a Beta distribution, making the model uncertainity-aware. Defaults to False.
        """
        
        super().__init__()
        
        self.use_beta = use_beta
        
        # initialze list of layers
        n_layers = len(dense)
        layers = []
        prev_dim = input_dim # first layer dimension = feature dimension
        # loop over layers
        for dim, act, drop in zip(dense, hidden_activation, dropout):
            # add type of layer
            layers.append(nn.Linear(prev_dim, dim))
            # add activation function
            layers.append(self._get_activation(act))
            # define dropout rate
            layers.append(nn.Dropout(drop))
            # store current dimension
            prev_dim = dim
        
        # common layers regarding of output style
        self.shared_layers = nn.Sequential(*layers)

        # add last layers
        # if use_beta -> last layer is dim 2 for alpha and beta parameters -> minimize beta_nll_loss
        if use_beta:
            self.alpha_head = nn.Linear(prev_dim, 1)
            self.beta_head = nn.Linear(prev_dim, 1)
        
        # if deterministic model -> minimize logloss 
        # Note: Sigmoid activation for binary output -> taken care of when using BCEWithLogitsLoss loss criterion
        else:
            self.output_head = nn.Linear(prev_dim, 1)
            #layers.append(nn.Linear(prev_dim, 1))
            #self.model = nn.Sequential(*layers)


    def forward(self, x):
        """
        Feed forward

        Args:
            x (pytorch.tensor): training data set.

        Returns:
            fitted model
        """
        # feed input through common layers
        h = self.shared_layers(x)
        # if fit Beta distribution, apply sofplus on last layer to force alpha and beta parameters to be positive
        # prior = 1e-3
        if self.use_beta:
            alpha = F.softplus(self.alpha_head(h)) + 1e-3 
            beta = F.softplus(self.beta_head(h)) + 1e-3 
            return alpha.squeeze(1), beta.squeeze(1) # make 1 dimensional
        else:
            return self.output_head(h).squeeze(1) # make 1 dimensional

    def _get_activation(self, act):
        return {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }.get(act, nn.ReLU())  # Default to ReLU

class TorchClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrapper around PyTorch model that handels Optuna pruning, early stopping on validation set and uncertainty-aware predictions 
    """

    def __init__(self, input_dim=None, dense=None, hidden_activation=None, dropout=None, lr=None, epochs=None, batch_size=None, 
                trial=None, pruning_threshold=0.05, pruning_min_epochs=10, score_for_pruning=accuracy_score, use_beta=False):
        """
        Define model. All undefined parameters are set to default in the _check_input function.
    
        Args:
            input_dim (int, optional): NN model input layer dimension (number of features). Defaults to None.
            dense (list, optional): NN architecture: list of integers. Defaults to None.
            hidden_activation (string, optional): global activation function. Defaults to None.
            dropout (float, optional): global dropout rate. Defaults to None.
            lr (float, optional): learning rate. Defaults to None.
            epochs (int, optional): max number of epoch. Defaults to None.
            batch_size (int, optional): batsh size for training. Defaults to None.
            trial (Optuna.trial, optional): Optuna trial object. Defaults to None.
            pruning_threshold (float, optional): threshold in score_for_prunning for pruning optuna trail. Defaults to 0.99 in accuracy (hard coded pruning).
            pruning_min_epochs (int, optional): minimum number of epoch to wait until pruning. Defaults to 10.
            score_for_pruning (function, optional): score to calculate performance for pruning. Defaults to accuracy_score.
            use_beta (bool, optional): if True, train the model to fit a Beta distribution, making the model uncertainty-aware. Defaults to False.
        """
        
        self.input_dim = input_dim
        self.dense = dense
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_activation = hidden_activation
        self.dropout = dropout

        # use Beta distribution as target
        self.use_beta = use_beta

        # need for sklearn compatibility: hard coded binary classification
        self.classes_ = np.array([0, 1])

        # placeholder for train and val loss per epoch/batch
        self.train_loss = []
        self.val_loss = []

        # optuna early pruning parameters
        self.trial = trial
        self.pruning_threshold = pruning_threshold
        self.pruning_min_epochs = pruning_min_epochs
        self.score_for_pruning = score_for_pruning

    def _build_model(self):
        """Builds the PyTorch model dynamically based on provided hyperparameters."""

        logger.info("Building PyTorch classifier ")

        # check model inputs
        self._check_inputs()

        # define NN
        self.model = FullyConnectedNN(self.input_dim, self.dense, self.hidden_activation, self.dropout, self.use_beta)
        # define optimizer 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # optimization criteria
        if self.use_beta:
            # -log_prob of a Beta distribution
            self.criterion = self._beta_nll_loss
            model_type = "uncertainity-aware"
        else:
            # standard logloss
            self.criterion = nn.BCEWithLogitsLoss()
            model_type = ""

        logger.success(f"Pytorch {model_type} model with input dimension {self.input_dim} built")

    def fit(self, X_in, y_in, X_val=None, y_val=None,  patience=20, tolerance=0.001):
        """
        Fit PyTorch model

        Args:
            X_in (array-like): Training data.
            y_in (array-like): Training boolean target.
            X_val (array-like, optional): Validation data. Defaults to None.
            y_val (array-like, optional): Validation boolean target. Defaults to None.
            patience and min_delta: addition arguments for early stopping. .

        Raises:
            ValueError:"Pytorch model input dimension mismatch: the number of features in fit differs from features in transform."
            exceptions.TrialPruned: returned Optuna's expected exception for pruning
        """
        
        # convert pandas to tensor
        X_tensor, y_tensor = self._convert_data(X_in, y_in) 

        # check and define input layer dimension based on given feature dimension
        if self.input_dim is None:
            self.input_dim = X_in.shape[1]
        
        # raise an error if the number of features differs (may happend when new features are created within the pipeline)
        elif X_in.shape[1] != self.input_dim:
            err_str = "Pytorch model input dimension mismatch: the number of features in fit differs from features in transform."
            logger.error(err_str)
            raise ValueError(err_str)

        # Setup validation for early stopping if validation step is provided
        X_val_tensor, y_val_tensor = self._convert_data(X_val, y_val) if X_val is not None and y_val is not None else (None, None)
        use_val = X_val_tensor is not None and y_val_tensor is not None
        if use_val:
            logger.info(f"Pytorch will use validation set for early stoping with patience={patience} and tolerance={tolerance}")
            early_stopper = EarlyStopper(patience=patience, tolerance=tolerance)
        
        # re-initialize train and val loss
        self.train_loss = []
        self.val_loss = []

        # build Pytorch model
        self._build_model()

        # check if attribute trial is not None (Optuna compatibility)
        trial = getattr(self, "trial", None)
        if trial is not None:
            logger.info("Checking for pruning optuna's trial")

        logger.info("Fitting PyTorch classifier: start ")

        # prepare batches
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # loop over epochs and batches for training
        for epoch in range(self.epochs):
            training_loss = 0.0
            val_loss = 0.0
            for batch_X, batch_y in loader:
                
                # Training 
                # ------------------------------------------------------

                # get optimization values
                self.optimizer.zero_grad()
                _, loss = self._get_outputs(batch_X, batch_y)
                loss.backward()
                self.optimizer.step()
                
                # store training loss
                training_loss += loss.item()

                # Validation
                # --------------------------------------------------------

               # loss on validation set for early stopping
                if use_val:
                    with torch.no_grad():
                        _, vloss = self._get_outputs(X_val_tensor, y_val_tensor)
                    val_loss += vloss.item()

            # take mean over batches for specific epochs
            val_loss = val_loss/len(loader)
            training_loss = training_loss/len(loader)
            # store training and validation loss per epochs
            self.train_loss.append(training_loss)
            self.val_loss.append(val_loss)

            # check early stopping
            if use_val: 
                if early_stopper(val_loss):
                    logger.info(f"Early stopping at epoch {epoch+1} (validation loss: {val_loss:.4f})")
                    break
            
            # if performing optuna tuning
            if trial is not None:

                # get score metric at given epochs
                pruning_score = self.score_for_pruning(y_tensor, self.predict(X_tensor))

                # manual prunning -> cancel trial if threshold on pruning_score is not reached
                if epoch >= self.pruning_min_epochs and pruning_score < self.pruning_threshold:
                    logger.warning(f"Trial pruned manually at epoch {epoch} with accuracy={pruning_score:.3f}")
                    raise exceptions.TrialPruned()

                # optuna prunning -> depending on define prunning method
                trial.report(pruning_score, step=epoch)
                if (trial.should_prune()):
                    logger.warning(f"Trial pruned from optuna at epoch {epoch} with accuracy={pruning_score:.3f}")
                    raise exceptions.TrialPruned()

        logger.success("Fitting PyTorch classifier: done ")

    def predict_proba(self, X_in):
        """
        Get raw probability output

        Args:
            X_in (array-like): data set to perform prediction.

        Returns:
            np.array: shape = (len(X_in), 2)
        """
        self.model.eval()
        
        if not self.use_beta:
            X_tensor, _ = self._convert_data(X_in)
            with torch.no_grad():
                logits = self.model(X_tensor).squeeze()
                p1 = torch.sigmoid(logits).numpy().reshape(-1, 1)  # convert logits to probabilities
                p0 = 1.0 - p1
            return np.hstack([p0, p1])

        # return mean of the Beta distribution from alpha and beta parameters
        else:
            mean = self.predict_uncert(X_in)["mu"].to_numpy()
            return np.hstack([(1 - mean).reshape(-1, 1), mean.reshape(-1, 1)])

    def predict(self, X_in, cutoff=0.5):
        """
        Simple threshold flagging

        Args:
            X_in (array-like): data set to perform prediction.
            cutoff (float, optional): Threshold for hard cut of the raw output probability. Defaults to 0.5.

        Returns:
            np.arry: predicted class
        """
        proba = self.predict_proba(X_in)[:, 1] 
        return (proba > cutoff).astype(int)
    
    def predict_uncert(self, X_in):
        """
        get the mean and variance of the Beta distribution for each data point

        Args:
            X_in (array-like): data set to perform prediction.

        Returns:
            pd.DataFrame: columns = mu and var.
        """
        if not self.use_beta:
            err_str = "Attempting to get mean and variance from a PyTorch model that is not uncertainty-aware. \
                        Re-define the model with the argument 'use_beta=True'"
            logger.error(err_str)
            raise ValueError(err_str)
            
        index_in = X_in.index
        X_tensor, _ = self._convert_data(X_in)
        with torch.no_grad():
            alpha, beta = self.model(X_tensor)
            mean = alpha / (alpha + beta)
            var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        return pd.DataFrame({'mu': mean, 'var': var}, index=index_in)

    def _get_outputs(self, X, y):
        """ get raw model output and loss"""
        if self.use_beta:
            alpha, beta = self.model(X)
            loss = self.criterion(y, alpha, beta)
            return (alpha, beta), loss
        else:
            outputs = self.model(X).squeeze()
            loss = self.criterion(outputs, y)
            return outputs, loss

    def _beta_nll_loss(self, y_true, alpha, beta, eps=1e-3):
        """Loss criterium for a Beta distribution target

        Args:
            y_true (int,float): flag. 
            alpha (int, float): predicted alpha value.
            beta (int, float): predicted beta value.
            eps (float, optional): avoid zero. Defaults to 1e-3.

        Returns:
            float: -log_prob of the Beta distribution
        """
        y_smooth = y_true * (1 - 2 * eps) + eps
        dist = torch.distributions.Beta(alpha, beta)
        return -dist.log_prob(y_smooth).mean()

    def _convert_data(self, X, y=None):
        """ convert input data to torch tensor """
        
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        elif not torch.is_tensor(X):
            raise ValueError(f"Feature data must be either a pandas DataFrame, a numpy array or toch tensors, got {X}") 

        if y is not None:
            y = y.to_numpy() if isinstance(y, pd.Series) else y
            y = torch.tensor(y, dtype=torch.float32)
        else:
            y = None
        return X, y

    def _check_inputs(self):
        """ check format input for the NN definition """
        
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

        # learning rate
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

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from loguru import logger
    import sys
    import pandas as pd

    logger.remove()
    logger.add(sys.stderr, level="SUCCESS")
    logger.add(sys.stderr, level="INFO")
    logger.add(sys.stderr, level="WARNING")
    logger.add(sys.stderr, level="ERROR")

    # random data
    dummy_X_data = pd.DataFrame(np.random.normal(1, 0.2, (1000, 10)))
    dummy_y_data = pd.Series(np.random.binomial(1,0.2, 1000))
    X_train, X_val, y_train, y_val = train_test_split(dummy_X_data, dummy_y_data, test_size=0.2)

    para = {"input_dim":10, 
            "dense":"5_5", 
            "hidden_activation":"relu", 
            "dropout":0.05, 
            "lr":0.01, 
            "epochs":5, 
            "batch_size":20}
    torch_model = TorchClassifier(**para)

    # fit
    torch_model.fit(X_train, y_train, X_val, y_val)

    # predict
    torch_predict = torch_model.predict_proba(X_train)

    # prediction
    print()
    print("Prediction probability")
    print("shape = ", torch_predict.shape)
    print("min = ", min(torch_predict[:,1]), "max = ", max(torch_predict[:,1]))
    print()

    # validation loss
    print("validation Vs training loss")
    df_loss = pd.DataFrame()
    df_loss["val_loss"] = torch_model.val_loss
    df_loss["training_loss"] = torch_model.train_loss
    print(df_loss)
    print()

    # With uncertainity
    torch_model = TorchClassifier(use_beta=True, **para)

    # fit
    torch_model.fit(X_train, y_train, X_val, y_val)

    # predict proba
    torch_predict = torch_model.predict_proba(X_train)
    
    # predict with uncertainity
    torch_predict_uncert = torch_model.predict_uncert(X_train)
    print("Uncertainity-aware model")
    print(torch_predict_uncert.head(10))
    print()

    print("Difference between raw probability output and Beta mean")
    print("MAE = ", round(np.mean(torch_predict-torch_predict_uncert)))
