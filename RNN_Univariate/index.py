import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras import callbacks

def spliting_univariate_data(dataset, start_idx, end_idx, past_size, forecast_size):
    input_data = []
    target_data = []
    start_idx = start_idx + past_size
    if(end_idx is None):
        end_idx = len(dataset) - forecast_size
    for i in range(start_idx, end_idx):
        indices = range(i-past_size, i)
        # 0, start_idx+past_size : 20개
        # 1, start_idx+past_size+1 : 20개
        # 2, start_idx+past_size+1 : 20개
        # ...
        # 830, 849 : 20개
        input_data.append(np.reshape(dataset[indices],(past_size, 1 )))
        # [[]], [[]], [[]], ... 계속 이어나감
        target_data.append(dataset[i+forecast_size])
        # [], [], [], ... 계속 이어나감
    return np.array(input_data), np.array(target_data)

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    markers = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if(delta):
        future = delta
    else:
        future = 0
    plt.title(title)
    for i, x in enumerate(plot_data):
        if(i):
            plt.plot(future, plot_data[i], markers[i], label=labels[i])
        else:
            plt.plot(time_steps,plot_data[i].flatten(), markers[i], label=labels[i])
    
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Steps')
    return plt

def show_loss_graph(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='loss')
    plt.plot(epochs, val_loss, 'r', label='val_loss')
    return plt

TRAIN_SPLIT = 790
# 790 : 하루 뒤, 800 : 이틀 뒤
PAST_SIZE = 20
FORECAST_SIZE = 0
# 0 : 하루 뒤, 1 : 이틀 뒤

# RNN Model data
BATCH_SIZE = 256
BUFFER_SIZE = 10000

df=pd.read_csv(r"./sungsan_all.csv")

print("\nversion: {}".format(tf.__version__))
# print(df.head())

uni_data = df['power(kWh)']
uni_data.index = df['datetime']
# print(uni_data.head())

# uni_data.plot(subplots=True,kind='line')
# plt.show()

# seed 값 설정. loss 율에 중요한 역할을 한다.
tf.random.set_seed(13)

# 표준화
uni_data = uni_data.values
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data - uni_train_mean) / uni_train_std
x_train_uni, y_train_uni = spliting_univariate_data(uni_data, 0, TRAIN_SPLIT, PAST_SIZE, FORECAST_SIZE)
x_val_uni, y_val_uni = spliting_univariate_data(uni_data, TRAIN_SPLIT, None, PAST_SIZE, FORECAST_SIZE)
# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample').show()

# 2차원 별로 numpy 배열 형식으로 slice
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BATCH_SIZE).batch(BUFFER_SIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BUFFER_SIZE).repeat()

# modeling
lstm_wind_forecast_model = tf.keras.models.Sequential()
lstm_wind_forecast_model.add(tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]))
lstm_wind_forecast_model.add(tf.keras.layers.Dense(1))
lstm_wind_forecast_model.compile(optimizer='adam', loss='mae')

# check pointing
# cp.wind 는 1일 뒤, cp.wind2 는 2일 뒤.
# 각각 1.05%, 1.01% 정도의 평균 오차율을 지니고 있다.
checkpoint_path = "./cp.wind"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
lstm_wind_forecast_model.load_weights(checkpoint_path)
# history = lstm_wind_forecast_model.fit(train_univariate, epochs=20, steps_per_epoch=200, validation_data=val_univariate, validation_steps=50, callbacks=[cp_callback])
# 과대적합을 막고싶다면, 더 많은 데이터를 가져와라.

# show_loss_graph(history).show()

all_value = 0

for x, y in val_univariate.take(1):
    # print(x)
    # print(lstm_wind_forecast_model(x)[0])
    for i in range(0,224):
        prediction = tf.get_static_value(lstm_wind_forecast_model.predict(x)[i][0])
        truth = tf.get_static_value(y[i])
        p = abs(prediction-truth)/100
        # print(p)
        all_value+=p
    
        #print(tf.get_static_value(lstm_wind_forecast_model(x)[i][0]))
        #plot = show_plot([x[i].numpy(), y[i].numpy(), lstm_wind_forecast_model.predict(x)[i]], 0, 'Wind Power Forecasting')
        #plot.show()

print("평균 오차율 : "+str(all_value)+" %")

'''
하루만큼의 예측을 알고싶으면,
[[[value 1],[value 2],[value 3],...,[value N]]] 을 predict 에 전달하여라.

여러일만큼의 예측을 알고싶으면,
[[[value 1],[value 2],[value 3],...,[value N]],[[value 1],[value 2],[value 3],...,[value N]]]
이런 식의 데이터를 predict 에 전달하여라.
'''