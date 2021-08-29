import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflowjs as tfjs


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


TRAIN_SPLIT = 800

PAST_SIZE = 20
FORECAST_SIZE = 0
STEP = 1

BATCH_SIZE = 128
BUFFER_SIZE = 10000

df = pd.read_csv("./sungsan_all.csv")
features_con = ['power(kWh)', 'speed(m/s)', 'pascal(hPa)']
features = df[features_con]
features.index = df['datetime']
features.plot(subplots=True)
# plt.show()
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean()
data_std = dataset[:TRAIN_SPLIT].std()
dataset = (dataset - data_mean) / data_std
print("mean: "+str(data_mean)+" std: "+str(data_std))

x_train_value, y_train_value = splitting_multivariate_data(
    dataset, dataset[:, 0], 0, TRAIN_SPLIT, PAST_SIZE, FORECAST_SIZE, STEP, single_step=True)
x_val_value, y_val_value = splitting_multivariate_data(
    dataset, dataset[:, 0], TRAIN_SPLIT, None, PAST_SIZE, FORECAST_SIZE, STEP, single_step=True)

# print(x_train_value[0].shape)
# print(y_train_value[0])

train_data_single = tf.data.Dataset.from_tensor_slices(
    (x_train_value, y_train_value))
train_data_single = train_data_single.cache().shuffle(10000).batch(128).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices(
    (x_val_value, y_val_value))
val_data_single = val_data_single.batch(128).repeat()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(
    32, input_shape=x_train_value.shape[-2:], activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop())
checkpoint_path = "./cp.wind.multiple"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# history = model.fit(train_data_single, epochs=80, steps_per_epoch=200,
#                     validation_data=val_data_single, validation_steps=50, callbacks=[early_stop, cp_callback])
model.load_weights(checkpoint_path)
# tfjs.converters.save_keras_model(model, "./")
all_percentage = 0
sample_length = 0
fp = open("./origin_sample.txt", "wt")
for x, y in val_data_single.take(3):
    prediction = model.predict(x)[0]
    truth = y[0]
    origin = x.numpy()
    for i in range(0, len(origin)):  # 한 라인
        fp.write("[")
        for j in range(0, len(origin[i])):
            fp.write("[")
            for k in range(0, 3):
                origin[i][j][k] = origin[i][j][k]*data_std+data_mean
                if(k == 2):
                    fp.write(str(float(str(origin[i][j][k]))))
                else:
                    fp.write(str(float(str(origin[i][j][k])))+", ")
            if(j == len(origin[i])-1):
                fp.write("]")
            else:
                fp.write("],")
        fp.write("],\n")

    # print(prediction)
    # print(truth)
    sample_length = len(x)
    for i in range(0, len(x)):
        prediction = model.predict(x)[i]
        truth = (y[i].numpy()*data_std)+data_mean
        prediction = prediction*data_std+data_mean
        p = abs(prediction - truth)/100
        all_percentage += p
        # print(str(i)+"번째 샘플 오차율: "+str(p)+"%")
        # plot = show_plot([x[i][:, 0].numpy(), y[i].numpy(),
        #                  model.predict(x)[i]], 24, 'Single Step Predcition')
        # plot.show()
    print("총 오차율 평균: "+str(all_percentage/len(x))+"%")
    all_percentage = 0
fp.close()
