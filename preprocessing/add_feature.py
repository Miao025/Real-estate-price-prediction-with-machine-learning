import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AddRegion(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        Flanders_provinces = ['Antwerp','East Flanders','Flemish Brabant','Limburg','West Flanders']
        Walloon_provinces = ['Hainaut','Liège','Namur','Luxembourg','Walloon Brabant']
        X['region'] = X['province'].map(lambda x: 'Flanders' if x in Flanders_provinces else ('Walloon' if x in Walloon_provinces else 'Brussels'))
        print('end of AddRegion')
        return X