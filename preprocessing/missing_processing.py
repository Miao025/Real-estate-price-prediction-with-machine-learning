import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ApartmentLandSurfaceTo0(BaseEstimator, TransformerMixin): # apartment landsurface should be 0
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X=X.copy()
        X.loc[X['type'] == 'apartment', 'landsurface'] = 0
        print('end of ApartmentLandSurfaceTo0')
        return X


class MissingToUnknown(BaseEstimator, TransformerMixin):
    def __init__(self, thres: float, only_to_hasXXX: bool):
        self.thres = thres
        self.only_to_hasXXX = only_to_hasXXX
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.only_to_hasXXX == True:
            cols = [col for col in list(X.columns) if col.startswith('has')]
        else:
            cols = []
            for col in X.columns.to_list():
                if X[col].isna().mean() >= self.thres:
                    cols.append(col)
        X[cols] = X[cols].fillna('unknown')
        print('end of MissingToUnknown')
        return X
    
class DropMissingCols(BaseEstimator, TransformerMixin):
    def __init__(self, thres: float, cols: list[str]):
        self.thres = thres
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.thres != None:
            X = X.loc[:, X.isnull().mean() <= self.thres]
        if self.cols != None:
            X = X.drop(columns=self.cols, axis=1)
        print('end of DropMissingCols')
        return X