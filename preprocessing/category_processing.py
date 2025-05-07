import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class EpcProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, ir_to_None: bool, label_encoding:bool):
        self.ir_to_None = ir_to_None
        self.label_encoding = label_encoding

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X=X.copy()
        if self.ir_to_None == True:
            X['epcScore']=X['epcScore'].map(lambda x: None if isinstance(x, str) and '_' in x else x) # replace G_C, E_C to None as they are meaningless!
        if self.label_encoding == True:
            order_epcScore = {'G': 0,
                                'F': 1,
                                'E': 2,
                                'D': 3,
                                'C': 4,
                                'B': 5,
                                'A': 6,
                                'A+': 7,
                                'A++': 8,
                                }
        X['epcScore_label_encoding'] = X['epcScore'].map(order_epcScore)
        X = X.drop(columns=['epcScore'])
        print('end of EpcProcessing')
        return X
    

class PostToGDP(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gdp_data = pd.read_csv('preprocessing/post_mapping.csv')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def get_gdp(code):
            for index, row in self.gdp_data.iterrows():
                if row['postCodeLower'] <= code <= row['postCodeUpper']:
                    return row['gdpPerCapita']
            return 54700 #set default to belgium average GDP per capita

        X['gdpPerCapita'] = X['postCode'].map(get_gdp)
        X = X.drop(columns=['postCode'])
        print("end of PostToGDP")
        return X
