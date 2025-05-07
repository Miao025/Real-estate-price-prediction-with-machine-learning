import h2o
from h2o.automl import H2OAutoML
from h2o.automl import get_leaderboard
import pandas as pd
from sklearn.model_selection import train_test_split

def autoML_h2o(X, y, top_n: int):
    # initiate h2o and get the train-test sets (note that h2o will handel cv automately in train set)
    h2o.init()
    df = pd.concat([X, y], axis=1)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_h2o = h2o.H2OFrame(train)
    test_h2o = h2o.H2OFrame(test)

    # set the param for automl and train it
    aml = H2OAutoML(
        # max_models=50, # try maximum 50 models
        max_runtime_secs=900, # limit total runtime to 900 seconds
        sort_metric='MAE', # use mae to rank models
        nfolds=5, # use 5-fold cross-validation
        stopping_metric='MAE', # early stopping based on mae
        stopping_rounds=30, # early stop after 30 rounds of no improvement
        seed=42 # random state 42
    )
    aml.train(x=[col for col in train_h2o.columns if col != 'price'], y='price', training_frame=train_h2o)

    # get top top_n models from the leaderboard
    lb = get_leaderboard(aml, extra_columns='ALL')
    top_models = lb.head(rows=top_n)

    # loop over top_n models to get their info
    for i in range(top_n):
        model_id = top_models.as_data_frame().iloc[i]['model_id']
        model = h2o.get_model(model_id)
        print(f'For top-{i+1} model: {model_id}:')

        # display the num of cols finally used
        print(f'Finally used features: {model.actual_params.get('predictor_columns', 'N/A')}')
            # model.actual_params is a dictionary that shows the actual values of parameters used in the trained model.
            # by .get('predictor_columns'), the list of features (col names) the model used during training is retrived.
            # however, when preprocessing changes the feature names, 'predictor_columns' might be none.
            # so set ]N/A' as the fallback value.

        # display pre-processing techniques, note that not all preprocessing steps are stored, some(e.g. imputation, encoding) are not.
        if 'preprocessing' in model._parms:
            print(f'with explicit preprocessing steps: {model._parms['preprocessing']}')

        # display hyperparam
        print("and hyperparameters:")
        for k, v in model.actual_params.items():
            print(f'{k}: {v}')
                    
        # info of test evaluation
        perf = model.model_performance(test_h2o)
        print(f'Test mae: {perf.mae():.2f}')
        print(f'Test R²: {perf.r2():.2f}')