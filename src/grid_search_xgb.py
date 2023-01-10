import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train_folds_updated.csv")

    X = df.drop(['total_cases', 'city', 'week_start_date', 'kfold'], axis=1).values
    y = df.total_cases.values
    regressor = xgb.XGBRegressor()
    param_grid = {
        "eta": [0.1, 0.3, 0.03, 0.003],
        'max_depth': [5, 7, 9],
        'min_child_weight': [3],
        'gamma': [0, 10, 100],
        'subsample': np.arange(0.1, 1.0, 0.1),
        'colsample_bytree': np.arange(0.1, 1.0, 0.1)
    }
    model = model_selection.GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        verbose=10,
        n_jobs=1,
        cv=3
    )
    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
