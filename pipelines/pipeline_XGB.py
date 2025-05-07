from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from preprocessing.add_feature import AddRegion 
from preprocessing.category_processing import EpcProcessing, PostToGDP
from preprocessing.missing_processing import ApartmentLandSurfaceTo0, MissingToUnknown, DropMissingCols
from preprocessing.format_dtype import ObjToCategory

def pipeline_XGB():
    pipeline = Pipeline([
        ('AddRegion', AddRegion()), # add variable 'region'
        ('EpcProcessing', EpcProcessing(ir_to_None=True, label_encoding=False)), # EPC E_D... to None, label encoding to 0,1,2...
        ('PostToGDP', PostToGDP()), # external feature mapping postCode to GDP per capita
        # ('DropMissingCols', DropMissingCols(thres=None, cols=['postCode'])), # drop the postCode col
        ('ApartmentLandSurfaceTo0', ApartmentLandSurfaceTo0()), # apartment should have 0 landsurface
        ('HighMissingToUnknown', MissingToUnknown(thres=0.3, only_to_hasXXX=False)), # if missing > 30%, change all missing values to 'unknown' and treat them as a group
        ('ObjToCategory', ObjToCategory()), # transfer obj type to category so that XGBoost can recognize
        ('model', XGBRegressor(objective='reg:absoluteerror', enable_categorical=True)) # use mae for loss function to reduce the influence of outliers
    ])
    print('end of build pipeline_XGB')
    return pipeline