import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib

import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle

from Multiple.LoadData import load_data
from Multiple import PlottingHelpers, ProcessingHelpers

importlib.reload(ProcessingHelpers) # while still working on than fun
importlib.reload(PlottingHelpers) # while still working on than fun

sns.set()

### Load Data into Workspace ###
dirname = os.getcwd()
pth = os.path.join(dirname, 'CMAPSSData')

print('loading data...')
dc = load_data(pth)
print('done')

# get the first data set training data
df = dc['FD_001']['df_train'].copy()

### Prepare Data ###
### Make a Column for the RUL target data (y) ###
# According to the data description document the data set contains multiple units,
# each unit starts at a certain degradation point and the measurement data ends
# closely before the unit was decommissioned of broke.
# Therefore assume, that for the last measurement time that is available for
# a unit the units RUL=0 (stopped measuring just before machine broke)
# get the time of the last available measurement for each unit
mapper = {}
for unit_nr in df['unit_nr'].unique():
    mapper[unit_nr] = df['time'].loc[df['unit_nr'] == unit_nr].max()

# calculate RUL = time.max() - time_now for each unit
df['RUL'] = df['unit_nr'].apply(lambda nr: mapper[nr]) - df['time']

### Drop the nan columns and rows ###
cols_nan = df.columns[df.isna().any()].tolist()
print('Columns with all nan: \n' + str(cols_nan) + '\n')

cols_const = [ col for col in df.columns if len(df[col].unique()) <= 2 ]
print('Columns with all const values: \n' + str(cols_const) + '\n')

df = df.drop(columns=cols_const + cols_nan)

### Perform an averaging with a floating average window of size 10 to smoothen out the signal noise ###
df_old = df.copy()
df = ProcessingHelpers.rolling_mean_by_unit(df, 10)
# Features by unit and RUL, before averaging
cols = [c for c in df.columns if c.startswith('s') or c in ['RUL', 'unit_nr']]
PlottingHelpers.plot_grouped_by_RUL(df_old[cols].copy())

# Features by unit and RUL, after averaging
cols = [c for c in df.columns if c.startswith('s') or c in ['RUL', 'unit_nr']]
PlottingHelpers.plot_grouped_by_RUL(df[cols].copy())

# Take out a certain percentage of units from the training data set for testing
units = df['unit_nr'].unique()
n_units = len(df['unit_nr'].unique())

units_test = random.sample(list(units), int(n_units * 0.2))
units_train = [nr for nr in units if nr not in units_test]

df_test = df.loc[df['unit_nr'].apply( lambda x: x in units_test )].copy()
df_train = df.loc[df['unit_nr'].apply( lambda x: x in units_train )].copy()

# Normalize the dataset by mean and std
f_exclude = ['sensor_09', 'sensor_14']
cols_features = [c for c in df_train.columns if c.startswith('s') and c not in f_exclude]

train_data = df_train[cols_features].values
train_labels = df_train['RUL'].values

test_data = df_test[cols_features].values
test_labels = df_test['RUL'].values

# Test data is *not* used when calculating the mean and std
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# First training sample, normalized
print(train_data[0])

### Define some helper functions ###
def plot_res(model, test_data, test_labels, train_data, train_labels):
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
    [loss, mae_train] = model.evaluate(train_data, train_labels, verbose=0)
    print("Testing set Mean Abs Error:  {:7.2f}".format(mae))
    print("Training set Mean Abs Error: {:7.2f}".format(mae_train))
    test_predictions = model.predict(test_data).flatten()
    train_predictions = model.predict(train_data).flatten()
    sns.distplot(train_predictions - train_labels)
    sns.distplot(test_predictions - test_labels)
    _ = plt.xlabel("Prediction Error")


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Validation loss')
    plt.legend()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

### Build tensorflow model ###
# our model consists of a simple 3 layered feedforward model with 24 nodes per layer.
# The output layer is a simple linear summing layer, which maps to the target values
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(24, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(24, activation=tf.nn.relu),
        keras.layers.Dense(24, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])

    return model

model = build_model()
model.summary()

### Train the model and validate it using the test data taken out before ###
# Also make the training process stop in case there is stagnation in the loss function of the validation data for more than 20 epochs
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

EPOCHS=500
train_data2, train_labels2 = shuffle(train_data, train_labels)
history = model.fit(train_data2, train_labels2, epochs=EPOCHS,
                    verbose=0,
                    validation_data = (test_data, test_labels),
                    callbacks=[early_stop, PrintDot()])

# Plot convergence over epochs
plot_history(history)

# That looks like the models are converging well. Lets take a look at the errors and their distributions.
### Plot prediciton error distibution ###
plot_res(model, test_data, test_labels, train_data, train_labels)

# Ok, the errors are more or less evenly distributed around zero.
# The with of the distribution seems pretty high, lets take a look
# at the predictions, actual values and errors over time.

### Plot the predicted RULs over the lifecycle of each unit, for the test data set ###
test_predictions = model.predict(test_data).flatten()
tmp = df_test[['RUL', 'unit_nr']]
tmp.loc[:,'RUL_error'] = test_predictions - test_labels
tmp.loc[:,'RUL_pred'] = test_predictions

PlottingHelpers.plot_grouped_by_RUL(tmp, leg=False, cols_data=['RUL', 'RUL_pred', 'RUL_error'])

### Plot the predicted RULs over the lifecycle of each unit, for the training data set ###
train_predictions = model.predict(train_data).flatten()
tmp = df_train[['RUL', 'unit_nr']]
tmp.loc[:,'RUL_error'] = train_predictions - train_labels
tmp.loc[:,'RUL_pred'] = train_predictions

PlottingHelpers.plot_grouped_by_RUL(tmp, leg=False, cols_data=['RUL', 'RUL_pred', 'RUL_error'])

# The predicted RULs are especially off at the beginning of the life cycle of a unit,
# but seem to get closer the more a machine nears to the end of it's life.
# This is especially valuable, when it comes to protecting the unit from damage,
# or scheduling maintenance closely to a breakdown.
#
# Looking at the data shown above when applying the windowed mean data,
# it seems intuitive, since the whole state of the units seem to stay
# relatively close to the comissioningstate for a long time and then starts
# deviating more the closer it comes the the end of its life cycle.

# Train another model but this time, let tensorflow take its own validation data based on 20% of the training data
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

model = build_model()
EPOCHS=500
train_data2, train_labels2 = shuffle(train_data, train_labels)
history = model.fit(train_data2, train_labels2, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
### Plot convergence over epochs ###
plot_history(history)
# That looks like the models are converging well. Lets take a look at the errors and their distributions.

### Plot prediciton error distibution ###
plot_res(model, test_data, test_labels, train_data, train_labels)
# Ok, the errors are more or less evenly distributed around zero.
# The width of the distribution seems pretty high,
# lets take a look at the predictions, actual values and errors over time.

### Plot the predicted RULs over the lifecycle of each unit, for the test data set ###
test_predictions = model.predict(test_data).flatten()
tmp = df_test[['RUL', 'unit_nr']]
tmp.loc[:,'RUL_error'] = test_predictions - test_labels
tmp.loc[:,'RUL_pred'] = test_predictions

PlottingHelpers.plot_grouped_by_RUL(tmp, leg=False, cols_data=['RUL', 'RUL_pred', 'RUL_error'])

### Plot the predicted RULs over the lifecycle of each unit, for the training data set ###
train_predictions = model.predict(train_data).flatten()
tmp = df_train[['RUL', 'unit_nr']]
tmp.loc[:,'RUL_error'] = train_predictions - train_labels
tmp.loc[:,'RUL_pred'] = train_predictions

PlottingHelpers.plot_grouped_by_RUL(tmp, leg=False, cols_data=['RUL', 'RUL_pred', 'RUL_error'])

### Results ###
# The predicited values for the trained models look ok, and get more reliable,
# the closer a unit nears the end of its lifecycle. Looking only at the last 30 cycles of the RUL shows

idx_test_end = test_labels < 30.0
idx_train_end = train_labels < 30.0

plot_res(model, test_data[idx_test_end], test_labels[idx_test_end], train_data[idx_train_end], train_labels[idx_train_end])

train_predictions = model.predict(train_data).flatten()
tmp = df_train[['RUL', 'unit_nr']]
tmp.loc[:,'RUL_error'] = train_predictions - train_labels
tmp.loc[:,'RUL_pred'] = train_predictions
tmp = tmp[tmp['RUL'] < 30.0]
PlottingHelpers.plot_grouped_by_RUL(tmp, leg=False, cols_data=['RUL', 'RUL_pred', 'RUL_error'])

# Which is much better, than what we saw above. This is good when it comes to protecting the unit from imminent
# danger and scheduling maintenance. This however can better be accomplished by modelling the problem as a
# classification problem. (See other notebooks)

### Summary ###
# The following things can be found:
#     The RUL estimation for RUL > 30 cycles is not very accurate, but reliable since it will get more accurate the smaller the actual RUL.
#     The RUL estimation for RUL <= 30 cycles is relatively accurate.
#     The units can be protected from breakdown and maintenenance/decomissioning can be sheduled safely some dozen cycles in advance before actual failure.
#     The imminent failure/protection problem is better modelled as a binary classification problem (see other notebooks)