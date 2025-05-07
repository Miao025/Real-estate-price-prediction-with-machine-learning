import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ObjToCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
            X = X.copy()
            obj_cols = X.select_dtypes(include=['object']).columns.tolist()
            X[obj_cols] = X[obj_cols].replace({True: 'True', False: 'False'})
            X[obj_cols] = X[obj_cols].astype('category')
            print('end of ObjToCategory')
            return X