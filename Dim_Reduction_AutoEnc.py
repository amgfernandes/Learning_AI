# Dimensionality Reduction using AutoEncoders

'''source: https://www.analyticsvidhya.com/blog/2021/06/dimensionality-reduction-using-autoencoders-in-python/ '''

# %%
'''Import libraries'''

''' Installation
conda create -n tensorflow python=3.8
conda activate tensorflow
conda install -c conda-forge tensorflow
conda install pandas
pip3 install -U scikit-learn
pip3 install keras-tuner
conda install seaborn    
'''

import math
import pandas as pd
import tensorflow as tf 
import kerastuner.tuners as kt
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.datasets import fetch_california_housing

# %%

california = fetch_california_housing(as_frame=True)

#%%
X = california.data
y = california.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

TARGET_NAME = y
# %%

'''Scale the dataset'''

from sklearn.preprocessing import MinMaxScaler

def scale_datasets(x_train, x_test):
  """
  Standard Scale test and train data
  """
  standard_scaler = MinMaxScaler()
  x_train_scaled = pd.DataFrame(
      standard_scaler.fit_transform(x_train),
      columns=x_train.columns
  )
  x_test_scaled = pd.DataFrame(
      standard_scaler.transform(x_test),
      columns = x_test.columns
  )
  return x_train_scaled, x_test_scaled
  
x_train_scaled, x_test_scaled = scale_datasets(x_train, x_test)
# %%
'''Train the autoencoder'''

class AutoEncoders(Model):
    
    def __init__(self, output_units):

        super().__init__()
        self.encoder = Sequential(
            [
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(7, activation="relu")
            ]
        )

        self.decoder = Sequential(
            [
            Dense(16, activation="relu"),
            Dense(32, activation="relu"),
            Dense(output_units, activation="sigmoid")
            ]
        )

    def call(self, inputs):
    
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

  
auto_encoder = AutoEncoders(len(x_train_scaled.columns))

auto_encoder.compile(
    loss='mae',
    metrics=['mae'],
    optimizer='adam'
)

history = auto_encoder.fit(
    x_train_scaled, 
    x_train_scaled, 
    epochs=15, 
    batch_size=32, 
    validation_data=(x_test_scaled, x_test_scaled)
)


# %%

'''Get the encoder layer and use the method predict to reduce dimensions in data. 
Since we have seven hidden units in the bottleneck the data is reduced to seven features.'''

encoder_layer = auto_encoder.get_layer('sequential')
reduced_df = pd.DataFrame(encoder_layer.predict(x_train_scaled))
reduced_df = reduced_df.add_prefix('feature_')
# %%

reduced_df 
# %%
