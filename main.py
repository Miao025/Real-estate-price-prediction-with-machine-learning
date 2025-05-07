from utils.load_data import load_and_clean_data
from utils.visualization import visualization
from pipelines.pipeline_XGB import pipeline_XGB
from models.train_cv_test_XGBoost import train_cv_test_XGBoost
from models.autoML_h2o import autoML_h2o

def main():
    X, y = load_and_clean_data('data/Kangaroo.csv')

    visualization(X, y, graph='boxplot')
    visualization(X, y, graph='scatter')

    pipeline = pipeline_XGB()
    train_cv_test_XGBoost(pipeline, X, y)

    autoML_h2o(X, y, top_n=1)

if __name__ == "__main__":
    main()