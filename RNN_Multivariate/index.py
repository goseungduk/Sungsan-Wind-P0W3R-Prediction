import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# RNN Model data
BATCH_SIZE = 256
BUFFER_SIZE = 10000

df = pd.read_csv(r"./sungsan_all.csv")

features_con = ['power(kWh)','speed(m/s)','pascal(hPa)']
features = df[features_con]
features.index = df['datetime']
print(features.head())
