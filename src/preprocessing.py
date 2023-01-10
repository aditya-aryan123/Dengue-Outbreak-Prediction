import pandas as pd
import numpy as np


def encode_cyclic_f(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_val)

    return data


df = pd.read_csv('../input/train_folds.csv')
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

df['month'] = df['week_start_date'].dt.month
df['dayofweek'] = df['week_start_date'].dt.dayofweek
df['dayofyear'] = df['week_start_date'].dt.dayofyear
df['quarter'] = df['week_start_date'].dt.quarter
df['season'] = df['month'] % 12 // 3 + 1
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x > 4 else 0)
df['is_weekday'] = df['dayofweek'].apply(lambda x: 1 if x < 4 else 0)

df = encode_cyclic_f(df, 'month', 12)
df = encode_cyclic_f(df, 'dayofweek', 6)
df = encode_cyclic_f(df, 'dayofyear', 358)
df = encode_cyclic_f(df, 'quarter', 4)
df = encode_cyclic_f(df, 'season', 4)

df.to_csv('updated_frame.csv', index=False)
