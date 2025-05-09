from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_cv_test_XGBoost(pipeline, X, y):
    print('XG Boost:')
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42) # get validation set for early-stopping when retrain on full training data
    # Note that in both XGB and LightGB algorithm, 'the CV will use the average scores from all folds, and use this for the early stopping' - see the founder's answer https://github.com/microsoft/LightGBM/pull/3204
    # therefore, we only use early stopping when retrain on full train-cv set.

    # get X_train and X_val transformed for further use
    transformer = Pipeline(pipeline.steps[:-1])  # Exclude estimator 'model'
    transformer.fit(X_train)
    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)

    # train + cross-validation: tuning hyper parameters with grid search
    param_grid = {
        'model__max_depth':[2, 4, 6, 8],
        'model__learning_rate':[0.05, 0.1, 0.2, 0.3]
    }
    grid_search = GridSearchCV(
        pipeline,
        param_grid = param_grid,
        cv=5, # 5-fold cv
        scoring='neg_mean_absolute_error', # attention: 1. neg is used as higher means better, 2. mse by default, but to reduce the influenc of outliers mae is used.
        n_jobs=-1, # use all computer cores, more speed!
        verbose=1 # how much info to print
    )
    grid_search.fit(
        X_train, y_train
        ) # get the best hyperparam
    print(f"Best hyperparameters: {grid_search.best_params_}")

    # retrain on full train-and-cv set
    pipeline.set_params(**grid_search.best_params_, model__early_stopping_rounds=30) # apply best hyperparam to pipeline, chosse 20–30 rounds if data is noisy, and if learning rate is small set the rounds higher
    pipeline.fit(
        X_train, y_train,
        model__eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)] # note that only the last set is used for early stopping
    ) # get param and the best num of trees
    best_model = pipeline # now the model is ready

    # extract metrics for plotting early stopping graph
    xgb_model = best_model.named_steps['model']  # access the model which holds the training results, including early stopping metrics
    eval_results = xgb_model.evals_result()  # retrive the logged dictionary of metrics(in our model: mae) for all the sets (train, validation)
    train_mae = eval_results['validation_0']['mae']  # get training mae
    val_mae = eval_results['validation_1']['mae']  # get validation mae
    iterations = range(len(train_mae))  # num of trees
    best_iteration = xgb_model.best_iteration  # early stopping point

    # plot early stopping graph
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_mae, label='Training MAE', color='blue')
    plt.plot(iterations, val_mae, label='Validation MAE', color='orange')
    plt.axvline(x=best_iteration, color='red', linestyle='--', label=f'Best Iteration ({best_iteration})')
    plt.xlabel('Iteration(num of trees)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('XGBoost Early Stopping: Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # display the num of cols finally used
    n_final_vars = X_train_transformed.shape[1]
    print(f'Finally used {n_final_vars} vars(cols)')

    # train evaluation
    y_pred = best_model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f'Train mae: {mae:.2f}')
    print(f'Train R²: {r2:.2f}')

    # validation evaluation
    y_pred = best_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'Validation mae: {mae:.2f}')
    print(f'Validation R²: {r2:.2f}')

    # test evaluation
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Test mae: {mae:.2f}')
    print(f'Test R²: {r2:.2f}')

    return best_model