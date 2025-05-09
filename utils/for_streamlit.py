from utils.load_data import load_and_clean_data
from pipelines.pipeline_XGB import pipeline_XGB
from models.train_cv_test_XGBoost import train_cv_test_XGBoost
import pickle

X, y = load_and_clean_data('data/Kangaroo.csv')
pipeline = pipeline_XGB()
model = train_cv_test_XGBoost(pipeline, X, y)

with open('streamlit_model.pkl', 'wb') as f:
    pickle.dump(model, f)