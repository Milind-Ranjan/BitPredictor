import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# --- DATA LOADING AND PREPARATION ---

def load_data(file_path="BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv"):
    """
    Loads and preprocesses the Bitcoin price data.
    """
    if not os.path.exists(file_path):
        print("Downloading data...")
        os.system(f"curl -L -o {file_path} https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv")
    
    df = pd.read_csv(
        file_path,
        parse_dates=["Date"],
        index_col=["Date"]
    )
    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()
    return timesteps, prices

def split_data(timesteps, prices, test_split=0.2):
    """
    Splits time series data into train and test sets.
    """
    split_size = int((1 - test_split) * len(prices))
    X_train, y_train = timesteps[:split_size], prices[:split_size]
    X_test, y_test = timesteps[split_size:], prices[split_size:]
    return X_train, y_train, X_test, y_test

# --- WINDOWING ---

def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    windowed_array = x[window_indexes]
    windows, labels = windowed_array[:, :-horizon], windowed_array[:, -horizon:]
    return windows, labels

def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split))
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels

# --- MODEL ---

def create_model(horizon, name="model_dense"):
    """
    Creates a simple Dense model.
    """
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(horizon, activation="linear")
    ], name=name)
    
    model.compile(loss="mae",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["mae"])
    return model

def make_preds(model, input_data):
  """
  Uses model to make predictions on input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast)

# --- EVALUATION ---

def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))
  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
  return mae / mae_naive_no_season

def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics
  mae = tf.keras.metrics.MeanAbsoluteError()(y_true, y_pred)
  mse = tf.keras.metrics.MeanSquaredError()(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.MeanAbsolutePercentageError()(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)
  
  if mae.ndim > 0:
      mae = tf.reduce_mean(mae)
      mse = tf.reduce_mean(mse)
      rmse = tf.reduce_mean(rmse)
      mape = tf.reduce_mean(mape)
      mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}

# --- VISUALIZATION ---

def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  """
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("BTC Price")
  if label:
    plt.legend(fontsize=14)
  plt.grid(True)


if __name__ == '__main__':
    # Set hyperparameters
    HORIZON = 1
    WINDOW_SIZE = 7

    # Load and split data
    timesteps, prices = load_data()
    X_train_full, y_train_full, X_test_full, y_test_full = split_data(timesteps, prices)

    # Create windowed data
    full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    
    # Create and train model
    model = create_model(HORIZON)
    print("Training model...")
    model.fit(x=train_windows,
              y=train_labels,
              epochs=100,
              verbose=1,
              batch_size=128,
              validation_data=(test_windows, test_labels))

    # Make predictions
    model_preds = make_preds(model, test_windows)

    # Evaluate predictions
    model_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_preds)
    print("\nModel Evaluation Results:")
    print(model_results)

    # Plot results
    print("\nDisplaying prediction results...")
    plt.figure(figsize=(10, 7))
    plot_time_series(timesteps=X_test_full[-len(test_windows):], values=test_labels[:, 0], label="Test_data")
    plot_time_series(timesteps=X_test_full[-len(test_windows):], values=model_preds, format="-", label="Model Predictions")
    plt.show() 