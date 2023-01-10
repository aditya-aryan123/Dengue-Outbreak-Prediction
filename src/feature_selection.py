from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from operator import itemgetter
import pandas as pd


def check_multicollinearity():
    df = pd.read_csv('../input/train_folds.csv')
    X = df.drop(['total_cases', 'week_start_date', 'city'], axis=1)
    X = add_constant(X)

    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_info['Columns'] = X.columns
    vif_info = vif_info.sort_values('VIF', ascending=False)
    vif_info.to_csv('variance_inflation_factor.csv', index=False)


def check_feat_imp():
    df = pd.read_csv('../input/train_folds.csv')
    X = df.drop(['total_cases', 'week_start_date', 'city'], axis=1)
    y = df['total_cases']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    model = DecisionTreeRegressor()
    rfe = RFE(estimator=model, n_features_to_select=0.8)
    rfe.fit(X_train, y_train)

    features = X_train.columns.to_list()
    for x, y in (sorted(zip(rfe.ranking_, features), key=itemgetter(0))):
        print(x, y)


print(check_multicollinearity())
print(check_feat_imp())
