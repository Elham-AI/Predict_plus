from pytorch_tabnet.tab_model import TabNetRegressor,TabNetClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pytorch_tabnet.callbacks import Callback
import os
import torch
from pytorch_tabnet.callbacks import Callback

class TabNetRegressorModel(BaseEstimator, RegressorMixin):
    def __init__(self,**kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        
    def fit(self, X, y):
        
        # Train TabNet model
        history = self.model.fit(
            X_train=X,
            y_train=y.reshape(-1, 1),
            max_epochs=500,
            patience=50,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        
        return self
    
    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()
    
    def __deepcopy__(self, memo):
        # Add deepcopy support for scikit-learn
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
class TabNetClassifierModel(BaseEstimator, RegressorMixin):
    def __init__(self,**kwargs):
        self.model = TabNetClassifier(**kwargs)
        self.kwargs = kwargs
        
    def fit(self, X, y):
        
        # Train TabNet model
        history = self.model.fit(
            X_train=X,
            y_train=y.reshape(-1, 1),
            max_epochs=500,
            patience=50,
            batch_size=256,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
        )
        
        return self
    
    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()
    
    def __deepcopy__(self, memo):
        # Add deepcopy support for scikit-learn
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result