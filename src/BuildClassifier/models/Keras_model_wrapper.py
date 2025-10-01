"""
Wrapper or Keras model using scikeras
Note: there is a compatibility isuue between Keras 3 and transformers, which needs to be downgraded ...
See for example: https://github.com/huggingface/transformers/issues/34761

Working version:

"""

from typing import Union

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Optimizer

from scikeras.wrappers import KerasClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F

class PickableKerasClassifier(KerasClassifier):
    def __getstate__(self):
        keras_sk_params = self.sk_params
        keras_classes_ = self.classes_
        keras_arch = self.model.to_json()
        keras_weights = self.model.get_weights()

        return dict(keras_sk_params=keras_sk_params,
                    keras_classes_=keras_classes_,
                    keras_arch=keras_arch,
                    keras_weights=keras_weights)

    def __setstate__(self, state):
        self.sk_params = state['keras_sk_params']
        self.classes_ = state['keras_classes_']
        self.model = model_from_json(state['keras_arch'])
        self.model.set_weights(state['keras_weights'])
        

def get_model(input_shape: int, dense: list, hidden_activation: list, dropout: list, optimizer: Union[Optimizer, str]):
    """
    Create fully connected network with given parameters.
    """
    # make sure layer metadata size matches
    assert len(dense) == len(hidden_activation) == len(dropout)
    # create sequential model
    model = Sequential()
    # add input layer
    model.add(Dense(dense[0], activation=hidden_activation[0], input_shape=input_shape))
    model.add(Dropout(dropout[0]))
    # add other layers
    for u, a, p in zip(dense[1:], hidden_activation[1:], dropout[1:]):
        model.add(Dense(u, activation=a))
        model.add(Dropout(p))
    # add output layer
    model.add(Dense(2, activation='softmax'))
    # compile model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

