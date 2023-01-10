import argparse
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

import model_dispatcher


def run(fold, model, model_type):
    if model_type == 'tree':

        df = pd.read_csv('updated_frame.csv')
        df.drop(['city', 'week_start_date'], axis=1, inplace=True)

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop('total_cases', axis=1)
        y_train = df_train.total_cases.values

        x_valid = df_valid.drop('total_cases', axis=1)
        y_valid = df_valid.total_cases.values

        clf = model_dispatcher.models[model]
        clf.fit(x_train, y_train)
        preds = clf.predict(x_valid)

        mse = metrics.mean_squared_error(y_valid, preds)
        rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
        mae = metrics.mean_absolute_error(y_valid, preds)
        r2 = metrics.r2_score(y_valid, preds)
        print(f"Fold={fold}, Root Mean Squared Error={rmse}, Mean Squared Error={mse}, Mean Absolute Error={mae},"
              f" R Squared={r2}")

    else:

        df = pd.read_csv('updated_frame.csv')
        df.drop(['city', 'week_start_date'], axis=1, inplace=True)

        '''cols = ['dayofyear', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2', 'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm', 'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
                'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c', 'station_min_temp_c',
                'station_precip_mm', 'year', 'weekofyear', 'total_cases', 'month', 'dayofweek', 'quarter', 'season',
                'quarter_cos', 'quarter_sin', 'season_cos', 'season_sin', 'season_sin']

        for col in cols:
            scaler = MinMaxScaler()
            df.loc[:, col] = scaler.fit_transform(df[col].values.reshape(-1, 1))'''

        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        x_train = df_train.drop(['total_cases', 'kfold'], axis=1)
        y_train = df_train.total_cases.values

        x_valid = df_valid.drop(['total_cases', 'kfold'], axis=1)
        y_valid = df_valid.total_cases.values

        clf = model_dispatcher.models[model]
        pipeline = Pipeline([('scaler', RobustScaler()), ('model', clf)])
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_valid)

        mse = metrics.mean_squared_error(y_valid, preds)
        rmse = metrics.mean_squared_error(y_valid, preds, squared=False)
        mae = metrics.mean_absolute_error(y_valid, preds)
        r2 = metrics.r2_score(y_valid, preds)
        print(f"Fold={fold}, Root Mean Squared Error={rmse}, Mean Squared Error={mse}, Mean Absolute Error={mae}, R "
              f"Squared={r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--model_type",
        type=str
    )
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model,
        model_type=args.model_type
    )
