import shap
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from yellowbrick.regressor import prediction_error


def train():
    df = pd.read_csv('../input/train_folds_updated.csv')
    df.drop(['kfold', 'week_start_date'], axis=1, inplace=True)
    test = pd.read_csv('../input/test_updated.csv')

    print(df.shape)
    print(test.shape)

    df['city'] = df['city'].replace({'sj': 1, 'iq': 2})
    test['city'] = test['city'].replace({'sj': 1, 'iq': 2})

    X = df.drop('total_cases', axis=1)
    y = df['total_cases']

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=None)

    model = xgb.XGBRegressor(eta=0.1, min_child_weight=3, max_depth=7, gamma=100, colsample_bytree=0.8, subsample=0.9)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, pred)
    rmse = metrics.mean_squared_error(y_test, pred, squared=False)
    mae = metrics.mean_absolute_error(y_test, pred)
    print(f"Root Mean Squared Error={rmse}, Mean Squared Error={mse}, Mean Absolute Error={mae}")

    visualizer = prediction_error(model, X_train, y_train, X_test, y_test)
    print(visualizer)

    '''feat_imp = model.get_booster().get_score(importance_type="gain")
    feat_imp_df = pd.DataFrame(feat_imp, index=[0])
    feat_imp_df.to_csv('feature_importance_xgboost.csv', index=False)'''

    '''explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, color=plt.get_cmap("tab20c"))'''
    test.drop('week_start_date', inplace=True, axis=1)
    test['prediction'] = model.predict(test)
    test['prediction'] = test['prediction'].astype('int64')
    # test = test.merge(test['prediction'], how='left', left_index=True, right_index=True)
    '''submission = test[['year', 'city', 'weekofyear', 'prediction']]
    submission.sort_values(by='year', ascending=False)
    submission.rename(columns={'prediction': 'total_cases'}, inplace=True)
    submission['city'].replace({1: 'sj', 2: 'iq'}, inplace=True)
    submission.to_csv('submission.csv', index=False)'''
    submission = test.copy()
    submission.sort_values(by='year', ascending=False)
    submission.rename(columns={'prediction': 'total_cases'}, inplace=True)
    submission['city'].replace({1: 'sj', 2: 'iq'}, inplace=True)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    train()
