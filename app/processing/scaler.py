import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class customScaler(BaseEstimator,TransformerMixin):
    def __init__(self,scaler:object):    
        self.columns = None

        if not isinstance(scaler, object):
            raise ValueError("variables should be a object")

        self.scaler = scaler

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = X.copy()
        self.columns = list(X.columns)
        self.scaler.fit(X)
        return self  

    def transform(self,X : pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(X)
        return pd.DataFrame(X,columns = self.columns)  
