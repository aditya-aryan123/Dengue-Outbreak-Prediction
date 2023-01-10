import xgboost as xgb
import lightgbm as lgb
import catboost as cgb
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model

models = {
    'linear_regression': linear_model.LinearRegression(),
    'lasso': linear_model.Lasso(alpha=1.5, max_iter=1000, random_state=1),
    'ridge': linear_model.Ridge(alpha=1.5),
    'decision_tree': tree.DecisionTreeRegressor(),
    'random_forest': ensemble.RandomForestRegressor(),
    'extra_tree': ensemble.ExtraTreesRegressor(),
    'xgboost': xgb.XGBRegressor(),
    'lightgbm': lgb.LGBMRegressor(),
    'catboost': cgb.CatBoostRegressor()
}
