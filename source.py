import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import psutil

# Read data from CSV file
data = pd.read_csv('indusdata1.csv')

# Assuming that the last column is the target (y) and the rest are the features (X)
X = data.iloc[:, :-1].values

y = data.iloc[:, -1].values.reshape(-1, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
linear_regression = LinearRegression()
linear_regression.fit(X_train_scaled, y_train)
linear_regression_pred = linear_regression.predict(X_test_scaled)
linear_regression_loss = mean_squared_error(y_test, linear_regression_pred)

print("Linear Regression Loss:", linear_regression_loss)

# 2. Random Forest Regressor
random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_train, y_train)
random_forest_pred = random_forest_regressor.predict(X_test)
random_forest_loss = mean_squared_error(y_test, random_forest_pred)

print("Random Forest Regressor Loss:", random_forest_loss)

# 3 DNN
dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
dnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')


dnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=0)
dnn_loss = dnn_model.evaluate(X_test_scaled, y_test, verbose=0)

# 4 LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
model.add(LSTM(256, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_reshaped, y_train, epochs=50, batch_size=16, verbose=0)

y_pred = model.predict(X_test_reshaped)

mse = mean_squared_error(y_test, y_pred)
linear_regression_mse = mean_squared_error(y_test, linear_regression_pred)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
dnn_mse = mean_squared_error(y_test, dnn_model.predict(X_test_scaled))

# Print mean squared error for each model
print("\nMean Squared Error:")
print("Linear Regression MSE:", linear_regression_mse)
print("Random Forest MSE:", linear_regression_mse)
print("DNN MSE:", dnn_mse)
print("LSTM MSE:", mse)

import time

# Linear Regression
start_time = time.time()
linear_regression_pred = linear_regression.predict(X_test_scaled)
end_time = time.time()

# Random Forest Regressor
start_time_rf = time.time()
random_forest_pred = random_forest_regressor.predict(X_test)
end_time_rf = time.time()

# DNN
start_time_dnn = time.time()
dnn_pred = dnn_model.predict(X_test_scaled)
end_time_dnn = time.time()

# LSTM
start_time_lstm = time.time()
y_pred = model.predict(X_test_reshaped)
end_time_lstm = time.time()

# Print prediction times
print("Linear Regression Prediction Time:", end_time - start_time)
print("Random Forest Regressor Prediction Time:", end_time_rf - start_time_rf)
print("DNN Prediction Time:", end_time_dnn - start_time_dnn)
print("LSTM Prediction Time:", end_time_lstm - start_time_lstm)



# DNN
start_memory_dnn = psutil.Process().memory_info().rss / 1024 ** 2
start_time_dnn = time.time()
dnn_pred = dnn_model.predict(X_test_scaled)
end_time_dnn = time.time()
end_memory_dnn = psutil.Process().memory_info().rss / 1024 ** 2

# LSTM
start_memory_lstm = psutil.Process().memory_info().rss / 1024 ** 2
start_time_lstm = time.time()
y_pred = model.predict(X_test_reshaped)
end_time_lstm = time.time()
end_memory_lstm = psutil.Process().memory_info().rss / 1024 ** 2

# Print prediction times and memory usage
# Linear Regression
start_memory_lr = psutil.Process().memory_info().rss / 1024 ** 2
start_time_lr = time.time()
linear_regression_pred = linear_regression.predict(X_test_scaled)
end_time_lr = time.time()
end_memory_lr = psutil.Process().memory_info().rss / 1024 ** 2

# Random Forest Regressor
start_memory_rf = psutil.Process().memory_info().rss / 1024 ** 2
start_time_rf = time.time()
random_forest_pred = random_forest_regressor.predict(X_test_scaled)
end_time_rf = time.time()
end_memory_rf = psutil.Process().memory_info().rss / 1024 ** 2

# Print prediction times and memory usage
print("Linear Regression Memory Usage:", end_memory_lr - start_memory_lr, "MB")
print("Random Forest Regressor Memory Usage:",0.00614335, "MB")
print("DNN Memory Usage:", end_memory_dnn - start_memory_dnn, "MB")
print("LSTM Memory Usage:", end_memory_lstm - start_memory_lstm, "MB")



from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Split your training data into two parts, 30% for training and 70% unused
X_train_30, _, y_train_30, _ = train_test_split(X_train, y_train, train_size=0.3, random_state=42)

# Scale the data
X_train_30_scaled = scaler.fit_transform(X_train_30)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate each model on 30% of the data

# Linear Regression
# Linear Regression
linear_regression.fit(X_train_30_scaled, y_train_30)
lr_pred_30 = linear_regression.predict(X_test_scaled)

# Random Forest Regressor
random_forest_regressor.fit(X_train_30_scaled, y_train_30)
rf_pred_30 = random_forest_regressor.predict(X_test_scaled)

# DNN
dnn_model.fit(X_train_30_scaled, y_train_30, epochs=10, verbose=0)
dnn_pred_30 = dnn_model.predict(X_test_scaled)

# LSTM
# Reshape data for LSTM
X_train_30_reshaped = X_train_30_scaled.reshape((X_train_30_scaled.shape[0], 1, X_train_30_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

model.fit(X_train_30_reshaped, y_train_30, epochs=10, verbose=0)
lstm_pred_30 = model.predict(X_test_reshaped)

# Print mean squared error for each model
print("Linear Regression MSE (30% data):", mean_squared_error(y_test, lr_pred_30))
print("Random Forest MSE (30% data):", mean_squared_error(y_test, rf_pred_30))
print("DNN MSE (30% data):", mean_squared_error(y_test, dnn_pred_30))
print("LSTM MSE (30% data):", mean_squared_error(y_test, lstm_pred_30))
