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


def splitting_multivariate_data(dataset, target, start_idx, end_idx,
                                history_size, target_size, step, single_step=False):
    data = []
    label = []
    start_idx = start_idx + history_size
    if(end_idx is None):
        end_idx = len(dataset) - target_size

    for i in range(start_idx, end_idx):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if (single_step):
            label.append(target[i+target_size])
        else:
            label.append(target[i:i+target_size])
    return np.array(data), np.array(label)


tf.random.set_seed(13)

# 전체데이터의 70%
TRAIN_SPLIT = 790

# 이전 30일의 데이터(24시간*20일) 기반으로 추적
# 미래 24시간 뒤의 데이터 예측
# 24시간 뒤의 데이터 보기
PAST_SIZE = 24*20
FORECAST_SIZE = 24*1
STEP = 1

# RNN Model data
BATCH_SIZE = 16
BUFFER_SIZE = 10000

df = pd.read_csv("./sungsan_all.csv")
features_con = ['power(kWh)', 'speed(m/s)', 'pascal(hPa)']
features = df[features_con]
features.index = df['datetime']

dataset = features.values
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)
scaled_df = pd.DataFrame(scaled)
scaled_df.columns = features_con

print(features.head())
print(scaled_df.head())
# inverse_scaled_data = scaler.inverse_transform(scaled.values)
# print(pd.DataFrame(inverse_scaled_data))
print(scaled[:, 0])
x_train_single, y_train_single = splitting_multivariate_data(
    scaled, scaled[:, 0], 0, TRAIN_SPLIT, PAST_SIZE, FORECAST_SIZE, STEP, single_step=True)
x_val_single, y_val_single = splitting_multivariate_data(
    scaled, scaled[:, 0], TRAIN_SPLIT, None, PAST_SIZE, FORECAST_SIZE, STEP, single_step=True)

# print(x_train_single[0].shape)

train_data_single = tf.data.Dataset.from_tensor_slices(
    (x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_single = tf.data.Dataset.from_tensor_slices(
    (x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(
    16, input_shape=x_train_single.shape[-2:], activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
checkpoint_path = "./cp.wind"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_data_single, epochs=10, steps_per_epoch=100,
                    validation_data=val_data_single, validation_steps=50, callbacks=[early_stop, cp_callback])
# model.load_weights("./cp.wind")
all_value = 0
for x, y in val_data_single.take(1):
    for i in range(0, 480):
        inversed_input = scaler.inverse_transform([[y[i], 0, 0]])
        truth = inversed_input[0][0]
        inversed_output = scaler.inverse_transform(
            [[model.predict(x[i])[0], 0, 0]])
        pred = inversed_output[0][0]
        p = abs(pred-truth)/100
        all_value += p
print(str(all_value)+"%")

# for i in range(0, 30):
#     pred = tf.get_static_value(model.predict(x)[i][0])
#     truth = tf.get_static_value(y[i])
#     p = abs(pred-truth)/100
#     print(p)
# print(x[6])
# print(model.predict(x)[5])
# plot = show_plot([x[5][:, 0].numpy(), y[5].numpy(),
#                   model.predict(x)[5]], 12,
#                  'Single Step Prediction')
# plot.show()
