# Real Estate Price Prediction with Machine Learning

A machine learning project to predict real estate prices in Belgium using advanced web scraping, data preprocessing, feature engineering, and model selection techniques. The project utilizes XGBoost and H2O AutoML for regression, and provides visualization and a Streamlit interface for interactive exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Visualization](#visualization)
- [Streamlit App](#streamlit-app)
- [Contributors](#contributors)
- [Timeline](#timeline)
- [Evaluation Criteria](#evaluation-criteria)
- [References](#references)

---

## Project Overview

This project aims to build a robust machine learning pipeline for predicting real estate prices in Belgium. It includes:

- Data scraping from Immoweb using a custom scraper
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation (XGBoost, H2O AutoML)
- Visualization of data
- Interactive Streamlit app for predictions

---

## Features

- **Data Scraping:**  
  A custom scraper ([`scraper.py`](data/scraper.py)) to extract property data from Immoweb.
- **Flexible Data Loading:** Easily load and clean datasets with [`load_and_clean_data`](utils/load_data.py).
- **Custom Preprocessing:** Modular transformers for missing values ([`DropMissingCols`](preprocessing/missing_processing.py), [`MissingToUnknown`](preprocessing/missing_processing.py)), categorical encoding ([`EpcProcessing`](preprocessing/category_processing.py)) and feature engineering ([`AddRegion`](preprocessing/add_feature.py), [`PostToGDP`](preprocessing/category_processing.py)(external feature mapping from third-party demographic data))
- **Model Pipelines:** Ready-to-use pipelines for XGBoost ([`pipeline_XGB`](pipelines/pipeline_XGB.py)).
- **Automated Model Selection:** H2O AutoML integration ([`autoML_h2o`](models/autoML_h2o.py)). This is an alternative to XGBoost model.
- **Comprehensive Evaluation:** Training, validation, and test metrics with early stopping visualization ([`train_cv_test_XGBoost`](models/train_cv_test_XGBoost.py)).
- **Visualization Tools:** Boxplots and scatter plots ([`visualization`](utils/visualization.py)).
- **Streamlit App:** Interactive app for predictions based on XGBoost model ([`display.py`](display.py)), also deployed [online ](https://real-estate-price-prediction-with-machine-learning-k2nt2ngadwc.streamlit.app/).

---

## Project Structure

```
.
├── main.py
├── display.py
├── model.pkl
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── example_data.csv
│   └── scraper.py
├── models/
│   ├── autoML_h2o.py
│   ├── train_cv_test_XGBoost.py
├── pipelines/
│   └── pipeline_XGB.py
├── preprocessing/
│   ├── add_feature.py
│   ├── category_processing.py
│   ├── format_dtype.py
│   ├── missing_processing.py
│   └── post_mapping.csv
├── utils/
│   ├── load_data.py
│   ├── save_to_pickle.py
│   └── visualization.py
```

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd Real-estate-price-prediction-with-machine-learning
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Scrape Data

- Use the scraper ([`scraper.py`](data/scraper.py)) to extract property data from Immoweb:
   ```sh
   python data/scraper.py
   ```
- Note: The [`example_data.csv`](data/example_data.csv) shows 3 example scraped entries.

### 2. Prepare Data

- Place your dataset (e.g., `data.csv`) in the `data/` directory.

### 3. Run Main Pipeline

- Edit [`main.py`](main.py) to set your dataset path.
- Run the main script:
   ```sh
   python main.py
   ```
- This will:
  - Load and clean data
  - Visualize data (boxplots, scatter plots)
  - Train and evaluate XGBoost and H2O AutoML models

### 4. Train and Save Model for Streamlit

- To train and save the ML model into pickle format for the Streamlit app:
   ```sh
   python utils/save_to_pickle.py
   ```
- This will save a trained model as `your_model.pkl` (An XGBoost model [`model.pkl`](model.pkl) is pre-trained for use).

### 5. Launch Streamlit App

- Start the Streamlit app locally for interactive predictions:
   ```sh
   streamlit run display.py
   ```
- Alternatively, access the deployed Streamlit app online [here](https://real-estate-price-prediction-with-machine-learning-k2nt2ngadwc.streamlit.app/).

---

## Data Preprocessing

- **Missing Value Handling:**  
  Custom transformers like [`DropMissingCols`](preprocessing/missing_processing.py) and [`MissingToUnknown`](preprocessing/missing_processing.py) handle missing values based on thresholds and column types.
- **Categorical Encoding:**  
  [`EpcProcessing`](preprocessing/category_processing.py) ordinal encodes EPC scores and handles invalid values.
- **Feature Engineering:**  
  Additional feature `Region` can be added via [`add_feature.py`](preprocessing/add_feature.py). [`PostToGDP`](preprocessing/category_processing.py) uses thrid-party GDP dataset [`post_mapping`](preprocessing/post_mapping.csv) to encode postal codes with corresponding GDP per capita values.

---

## Model Training & Evaluation

- **XGBoost Pipeline:**  
  Defined in [`pipeline_XGB`](pipelines/pipeline_XGB.py), using MAE as the loss function.
- **Training & Cross-Validation:**  
  [`train_cv_test_XGBoost`](models/train_cv_test_XGBoost.py) handles training, cross-validation with grid-search, early stopping with visualization, and evaluation (MAE, R²).
- **AutoML:**  
  [`autoML_h2o`](models/autoML_h2o.py) runs H2O AutoML, ranks models by MAE, reports top models' features and hyperparameters, and handels evaluation (MAE, R²).

---

## Visualization

- **Boxplots & Scatter Plots:**  
  Explore feature distributions and relationships of categorial and numerical variables to property price, respectively in boxplots and scatter plots.
- **Early Stopping Graphs:**  
  Plot training and validation MAE over boosting rounds for XGBoost.

---

## Streamlit App

- **Model Loading:**  
  Loads the model from `your_model.pkl` generated by [`save_to_pickle.py`](utils/save_to_pickle.py).
- **Interactive Predictions:**  
  Run Streamlit app [`display.py`](display.py) locally to input property features and get price predictions using the trained model.
- **Deployed App:**  
  Alternatively, access the deployed Streamlit app [online](https://real-estate-price-prediction-with-machine-learning-k2nt2ngadwc.streamlit.app/).

---

## Contributor

- Miao

---

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [pandas Documentation](https://pandas.pydata.org/docs/)