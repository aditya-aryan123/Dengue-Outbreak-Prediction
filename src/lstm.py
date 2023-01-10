import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping


df = pd.read_csv('updated_frame.csv')

df['city'] = df['city'].replace({'sj': 1, 'iq': 2})
df.drop('week_start_date', axis=1, inplace=True)


def df_to_x_y(df, window_size=1):
    X = []
    y = []
    for i in range(len(df) - window_size):
        row = [[a] for a in df[i: i + window_size, 0]]
        X.append(row)
        label = df[i + window_size, 0]
        y.append(label)
    return np.array(X), np.array(y)


train = df.loc[df['year'] < 2005]
test = df.loc[df['year'] > 2005]

scaler = StandardScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

time_steps = 12
X_train, y_train = df_to_x_y(scaled_train, time_steps)
X_test, y_test = df_to_x_y(scaled_test, time_steps)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, shuffle=False)

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True,
)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(loc="best")
plt.xlabel("No. Of Epochs")
plt.ylabel("mse score")
plt.show()
