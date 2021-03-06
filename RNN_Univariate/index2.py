import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(
            ), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def multivariate_data(dataset, target, start_idx, end_idx,
                      history_size, target_size, step, single_step=False):
    data = []
    label = []
    start_idx = start_idx + history_size
    if(end_idx is None):
        end_idx = len(dataset) - target_size

    for i in range(start_idx, end_idx):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if(single_step):
            label.append(target[i+target_size])
        else:
            label.append(target[i:i+target_size])
    return np.array(data), np.array(label)


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


tf.random.set_seed(13)

TRAIN_SPLIT = 60000

PAST_SIZE = 24*20
FORECAST_SIZE = 24 * 1
STEP = 1

df = pd.read_csv("./jeju_all.csv")
features_con = ['power(kWh)']
features = df[features_con]
features.index = df['datetime']

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean()
data_std = dataset[:TRAIN_SPLIT].std()
print("mean: "+str(data_mean))
print("std: "+str(data_std))
dataset = (dataset - data_mean) / data_std
x_train_multi, y_train_multi = multivariate_data(
    dataset, dataset[:, 0], 0, TRAIN_SPLIT, PAST_SIZE, FORECAST_SIZE, STEP)
x_val_multi, y_val_multi = multivariate_data(
    dataset, dataset[:, 0], TRAIN_SPLIT, None, PAST_SIZE, FORECAST_SIZE, STEP)

print("Single window of past history: {}".format(x_train_multi[0].shape))
print("Target Power to Predict: {}".format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices(
    (x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(10000).batch(256).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(256).repeat()

for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))
# True Points
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(24))
multi_step_model.compile(
    optimizer='adam', loss='mae')

# for x, y in val_data_multi.take(1):
#     print(multi_step_model.predict(x).shape)

history = multi_step_model.fit(train_data_multi, epochs=10,
                               steps_per_epoch=200, validation_data=val_data_multi, validation_steps=50)
