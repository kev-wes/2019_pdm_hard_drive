from datetime import date
from backblaze.load_data import LoadData
import pandas as pd
import random
from sklearn import pipeline
from sklearn.externals import joblib
import sklearn as sk
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn import svm
from math import sqrt
from timeit import default_timer as timer
from lifelines import CoxPHFitter
import numpy as np
from lifelines.utils import qth_survival_times
import time

### SOF Helper Functions ###
def rms(num):
    return sqrt(sum(n * n for n in num) / len(num))

def R_read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data
### EOF Helper Functions ###

if __name__ == "__main__":
    ### SOF Parameters ###
    ## Global
    data_path = "C:/Users/kwesendrup/PycharmProjects/master/backblaze/data/rawdata/"
    cols_no_data = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'RUL']
    #filtered_models = ['ST4000DM000', 'ST12000NM0007', 'ST8000NM0055', 'ST8000DM002', 'HGST HMS5C4040BLE640']
    # Filtered models after preprocessing
    filtered_models = [ 'ST4000DX000',
                        'ST6000DX000',
                        'ST8000DM002',
                        'ST8000NM0055',
                        'ST10000NM0086',
                        'ST12000NM0007',
                        'ST3160318AS',
                        'WDC WD30EFRX',
                        'WDC WD60EFRX',
                        'TOSHIBA MQ01ABF050',
                        'ST4000DM000',
                        'T',
                        'H',
                        'W',
                        'ST']
    ## Init
    # Set this to false to skip csv reading and load pickle if it was already saved
    initialize_data_set = False
    if initialize_data_set:
        start_date = date(2016, 1, 1)
        end_date = date(2017, 12, 31)
    ## Preprocessing
    # Set this to false to skip preprocessing and load pickle if it was already saved
    preprocess_data_set = False
    if preprocess_data_set:
        # Reduce sample size. Choose value in ]0, 1]
        sample_size = 1
        # Remove hard drives that did not have measurements in the last n days. 0 does not remove any.
        measurements_in_last_n_days = 0
        # Threshold for NaN values. If threshold_nan percent values are non-NaN, the column is not dropped
        threshold_nan = 0.95
        # Removes measurements before degradation is occuring if True
        elbow_point_detection = True
        # Rolling window size
        rolling_window = 10
    ## Train / test split
    # Set this to false to skip model training and load pickle if it was already saved
    train_test_split = False
    if train_test_split:
        test_size = 0.2
    ## Train and pred model
    # Set this to false to skip model training and load pickle if it was already saved
    svm = False
    cox = True
    # How to score
    scoring = sk.metrics.mean_squared_error
    # metrics.explained_variance_score(y_true, y_pred) 	Explained variance regression score function
    # metrics.max_error(y_true, y_pred) 	max_error metric calculates the maximum residual error.
    # metrics.mean_absolute_error(y_true, y_pred) 	Mean absolute error regression loss
    # metrics.mean_squared_error(y_true, y_pred[, …]) 	Mean squared error regression loss
    # metrics.mean_squared_log_error(y_true, y_pred) 	Mean squared logarithmic error regression loss
    # metrics.median_absolute_error(y_true, y_pred) 	Median absolute error regression loss
    # metrics.r2_score(y_true, y_pred[, …]) 	R^2 (coefficient of determination) regression score function.
    # k-fold for cross validation
    k_cv = 5
    ### EOF Parameters ###
    start_total = timer()

    ### SOF Initialize data set ###
    if initialize_data_set:
        start = timer()
        # Get failed entries
        data = LoadData(path = data_path,
                        start_date = start_date,
                        end_date = end_date,
                        failure = 1,
                        )
        failed_serials = data['serial_number'].unique().tolist()

        # Get all entries from failed drives w/ serial number
        data = LoadData(path = data_path,
                        start_date = start_date,
                        end_date = end_date,
                        serial = failed_serials
                        )
        data = data.sort_values(['serial_number', 'date'], ascending=True)
        # Save final dataframe as pickle
        data.to_pickle (data_path + "data_raw.pkl")
        end = timer()
        print("Raw dataset initialized and saved!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
              "Task time: " + str(round(end - start, 2)) + "\n" +
              "Total time: " + str(round(end - start_total, 2)))

    ### EOF Initialize data set ###
    if not filtered_models:
        data = pd.read_pickle(data_path + "data_raw.pkl")
        filtered_models = data['model'].unique()
    for model in filtered_models:
        ### SOF Data Preprocessing ###
        print("STARTING MODEL " + model)
        start = timer()
        data = pd.read_pickle(data_path + "data_raw.pkl")
        end = timer()
        print("Raw dataset loaded!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
              "Task time: " + str(round(end - start, 2)) + "\n" +
              "Total time: " + str(round(end - start_total, 2)))
        if preprocess_data_set:
            start = timer()
            # Filter models
            print("Start of data preprocessing" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
            data = data[data['model'].str.startswith(model)].copy()
            if data.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
            end = timer()
            print("After model reduction" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            start = timer()
            # only 21 columns are not pure NaN or constant per model
            # Backblaze suggests only to use:
            # - "smart_5_raw": "Reallocated_Sector_Count",
            # - "smart_187_raw": "Reported_Uncorrectable_Errors",
            # - "smart_188_raw": "Command_Timeout",
            # - "smart_197_raw": "Current_Pending_Sector_Count",
            # - "smart_198_raw": "Offline_Uncorrectable"
            # Source: https://www.backblaze.com/blog/hard-drive-smart-stats/
            # -"smart_9_raw": "Days_In_Service" is also used

            data = data.rename(index=str, columns={"smart_5_raw": "Reallocated_Sector_Count",
                                                   "smart_5_normalized": "Reallocated_Sector_Count_norm",
                                                   "smart_9_raw": "Days_In_Service",
                                                   "smart_9_normalized": "Days_In_Service_norm",
                                                   "smart_10_raw": "Spin_Retry_Count",
                                                   "smart_10_normalized": "Spin_Retry_Count_norm",
                                                   "smart_184_raw": "E2E_Error",
                                                   "smart_184_normalized": "E2E_Error_norm",
                                                   "smart_187_raw": "Reported_Uncorrectable_Errors",
                                                   "smart_187_normalized": "Reported_Uncorrectable_Errors_norm",
                                                   "smart_188_raw": "Command_Timeout",
                                                   "smart_188_normalized": "Command_Timeout_norm",
                                                   "smart_196_raw": "Reallocation_Event_Count",
                                                   "smart_196_normalized": "Reallocation_Event_Count_norm",
                                                   "smart_197_raw": "Current_Pending_Sector_Count",
                                                   "smart_197_normalized": "Current_Pending_Sector_Count_norm",
                                                   "smart_198_raw": "Offline_Uncorrectable",
                                                   "smart_198_normalized": "Offline_Uncorrectable_norm",
                                                   "smart_201_raw": "Soft_Read_Error_Rate",
                                                   "smart_201_normalized": "Soft_Read_Error_Rate_norm"}).copy()
            # Remove normalized values
            data.drop(list(data.filter(regex='normalized')), axis=1, inplace=True)
            #data = data[['date', 'serial_number', 'model', 'capacity_bytes', 'failure',
            #                    'Reallocated_Sector_Count', 'Days_In_Service', 'Reported_Uncorrectable_Errors',
            #                   'Command_Timeout', 'Current_Pending_Sector_Count', 'Offline_Uncorrectable']].copy()
            data = data.sort_values(['serial_number', 'date'], ascending=[True, False])
            # Assign RUL by counting upwards per serial number in the descended data frame
            data['RUL'] = data.groupby((data['serial_number'] !=
                                                      data['serial_number'].shift(1)).cumsum()).cumcount()
            data = data.sort_values(['serial_number', 'date'], ascending=True)
            # Divide hours in service by 24 to get days of service
            data[['Days_In_Service']] = data[['Days_In_Service']].div(24).round(0)
            # Remove entries where RUL = 0 and Days in Service = 0
            data = data[-((data.Days_In_Service == 0) &
                                        (data.RUL == 0))]
            end = timer()
            print("After pruning features and adding RUL" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # DEPRECATED Remove hard drives that have no measurements in last 10 days
            # Newly identified critical columns must be added to the check
            '''if measurements_in_last_n_days != 0:
                start = timer()
                df_lt_10 = data.loc[data['RUL'] <= measurements_in_last_n_days].copy()
                df_lt_10 = df_lt_10.loc[(df_lt_10['Reallocated_Sector_Count'] != 0) |
                                     (df_lt_10['Reported_Uncorrectable_Errors'] != 0) |
                                     (df_lt_10['Command_Timeout'] != 0) |
                                     (df_lt_10['Current_Pending_Sector_Count'] != 0) |
                                     (df_lt_10['Offline_Uncorrectable'] != 0)].copy()
                data = data[data['serial_number'].isin(df_lt_10['serial_number'].unique().tolist())].copy()
                del df_lt_10
                end = timer()
                print("After pruning hard drives w/o non-zero values in last " + str(measurements_in_last_n_days) + " days" + "\n" +
                     "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                    "Task time: " + str(round(end - start, 2)) + "\n" +
                    "Total time: " + str(round(end - start_total, 2)))'''

            # Min Max Normalization
            # Loop only over data columns
            start = timer()
            for feature_name in list(set(data.columns.values)- set(cols_no_data) - set(['Days_In_Service'])):
                max_value = data[feature_name].max()
                min_value = data[feature_name].min()
                data[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
            end = timer()
            print("After normalization" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # Drop NaN values
            start = timer()
            print("NaNs:\n" + str(data.isnull().sum(axis = 0)))
            # Drop columns if all NaN
            data = data.dropna(thresh=len(data.index)*threshold_nan,axis=1, how='all')
            if data.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
            end = timer()
            print("After dropping NaN columns" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # Drop columns if all constant
            start = timer()
            cols_const = [col for col in data.columns if len(data[col].unique()) <= 1]
            data = data.drop(data[cols_const], axis=1).copy()
            if data.size==0 or 'serial_number' not in data.columns: print("Model " + model + " skipped due to zero or one sample size."); continue
            end = timer()
            print("After dropping constant columns" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # Drop rows if any NaN
            # TODO Mean values of previous and next timestamp
            # sklearn.preprocessing.Imputer for NaN imputation
            start = timer()
            print("NaNs:\n" + str(data.isnull().sum(axis=0)))
            data = data.dropna()
            if data.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
            end = timer()
            print("After dropping NaN rows" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))

            # Feature Engineering
            # Zhao, R., Yan, R., Wang, J., and Mao, K. 2017. “Learning to Monitor Machine Health with
            # Convolutional Bi-Directional LSTM Networks,” Sensors, (17:2), p. 273.
            if rolling_window > 0:
                start = timer()
                for feature_name in list(set(data.columns.values) - set(cols_no_data)):
                    # Root mean square
                    data[feature_name + '_rms'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).apply(rms)).reset_index(level=0, drop=True)
                    # Peak-to-peak
                    data[feature_name + '_p2p'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).max()-x[feature_name].rolling(5).min()).reset_index(level=0, drop=True)
                    # Min
                    data[feature_name + '_min'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).min()).reset_index(level=0, drop=True)
                    # Max
                    data[feature_name + '_max'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).max()).reset_index(level=0, drop=True)
                    # Kurtosis
                    data[feature_name + '_kurt'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).kurt()).reset_index(level=0, drop=True)
                    data[feature_name + '_kurt'].fillna(0, inplace=True)
                    # Skewness
                    data[feature_name + '_skew'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).skew()).reset_index(level=0, drop=True)
                    data[feature_name + '_skew'].fillna(0, inplace=True)
                    # Mean
                    data[feature_name + '_mean'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).mean()).reset_index(level=0,drop=True)
                    # Variance
                    data[feature_name + '_var'] = data.groupby(['serial_number']).apply(
                        lambda x: x[feature_name].rolling(rolling_window).var()).reset_index(level=0, drop=True)
                end = timer()
                print("After feature engineering" + "\n" +
                      "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape)+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
                # Drop NaNs after sliding window
                start = timer()
                data = data.dropna()
                if data.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
                end = timer()
                print("After dropping NaN caused by feature engineering" + "\n" +
                      "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape)+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
                # Drop columns if all constant
                start = timer()
                cols_const = [col for col in data.columns if len(data[col].unique()) <= 1]
                data = data.drop(data[cols_const], axis=1).copy()
                if data.size==0 or 'serial_number' not in data.columns: print("Model " + model + " skipped due to zero or one sample size."); continue
                end = timer()
                print("After dropping constant columns caused by feature engineering" + "\n" +
                      "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                      "Task time: " + str(round(end - start, 2)) + "\n" +
                      "Total time: " + str(round(end - start_total, 2)))
            if 0 < sample_size < 1:
                start = timer()
                # Reduce sample size
                serials_filter = random.sample(list(data['serial_number'].unique()), int(len(data['serial_number'].unique()) * sample_size))
                data = data.loc[data['serial_number'].apply(lambda x: x in serials_filter)].copy()
                if data.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
                end = timer()
                print("After sample size reduction" + "\n" +
                      "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape)+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # Remove rows before first non-zero measurement of critical SMART values
            # "Elbow point detection"
            if elbow_point_detection:
                start = timer()
                # Are the critical values even recorded after data preprocessing?
                if not data.columns.isin(['Reallocated_Sector_Count', 'Reallocated_Sector_Count_norm', 'Spin_Retry_Count', 'Spin_Retry_Count_norm',
                    'E2E_Error', 'E2E_Error_norm', 'Reported_Uncorrectable_Errors', 'Reported_Uncorrectable_Errors_norm',
                    'Command_Timeout', 'Command_Timeout_norm', 'Reallocation_Event_Count', 'Reallocation_Event_Count_norm',
                    'Current_Pending_Sector_Count', 'Current_Pending_Sector_Count_norm', 'Offline_Uncorrectable',
                    'Offline_Uncorrectable_norm', 'Soft_Read_Error_Rate', 'Soft_Read_Error_Rate_norm']).any():
                    print("Model " + model + " skipped, because no critical measurements were recorded.")
                    continue
                df2 = pd.DataFrame()
                df = pd.DataFrame()
                for ser in data['serial_number'].unique():
                    df = data[data['serial_number'] == ser].copy()
                    df = df[('Reallocated_Sector_Count' in df.columns and (df.Reallocated_Sector_Count != 0).cumsum() > 0) |
                            ('Reallocated_Sector_Count_norm' in df.columns and (df.Reallocated_Sector_Count_norm != 1).cumsum() > 0) |
                            ('Spin_Retry_Count' in df.columns and (df.Spin_Retry_Count != 0).cumsum() > 0) |
                            ('Spin_Retry_Count_norm' in df.columns and (df.Spin_Retry_Count_norm != 1).cumsum() > 0) |
                            ('E2E_Error' in df.columns and (df.E2E_Error != 0).cumsum() > 0) |
                            ('E2E_Error_norm' in df.columns and (df.E2E_Error_norm != 1).cumsum() > 0) |
                            ('Reported_Uncorrectable_Errors' in df.columns and (df.Reported_Uncorrectable_Errors != 0).cumsum() > 0) |
                            ('Reported_Uncorrectable_Errors_norm' in df.columns and (df.Reported_Uncorrectable_Errors_norm != 1).cumsum() > 0) |
                            ('Command_Timeout' in df.columns and (df.Command_Timeout != 0).cumsum() > 0) |
                            ('Command_Timeout_norm' in df.columns and (df.Command_Timeout_norm != 1).cumsum() > 0) |
                            ('Reallocation_Event_Count' in df.columns and (df.Reallocation_Event_Count != 0).cumsum() > 0) |
                            ('Reallocation_Event_Count_norm' in df.columns and (df.Reallocation_Event_Count_norm != 1).cumsum() > 0) |
                            ('Current_Pending_Sector_Count' in df.columns and (df.Current_Pending_Sector_Count != 0).cumsum() > 0) |
                            ('Current_Pending_Sector_Count_norm' in df.columns and (df.Current_Pending_Sector_Count_norm != 1).cumsum() > 0) |
                            ('Offline_Uncorrectable' in df.columns and (df.Offline_Uncorrectable != 0).cumsum() > 0) |
                            ('Offline_Uncorrectable_norm' in df.columns and (df.Offline_Uncorrectable_norm != 1).cumsum() > 0) |
                            ('Soft_Read_Error_Rate' in df.columns and (df.Soft_Read_Error_Rate != 0).cumsum() > 0) |
                            ('Soft_Read_Error_Rate_norm' in df.columns and (df.Soft_Read_Error_Rate_norm != 1).cumsum() > 0)]
                    df2 = df2.append(df)
                del df
                data = df2.copy()
                del df2
                # Drop columns if all constant
                cols_const = [col for col in data.columns if len(data[col].unique()) <= 1]
                data = data.drop(data[cols_const], axis=1).copy()
                if data.size==0 or 'serial_number' not in data.columns: print("Model " + model + " skipped due to zero or one sample size."); continue

                end = timer()
                print("After removal of rows before elbow point" + "\n" +
                        "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape)+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            start = timer()
            data.to_pickle(data_path + "data_preprocessed_" + model + ".pkl")
            end = timer()
            print("Dataset preprocessed and saved!" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape)+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
        elif train_test_split:
            start = timer()
            data = pd.read_pickle(data_path + "data_preprocessed_" + model + ".pkl")
            end = timer()
            print("Preprocessed dataset loaded!" + "\n" +
                  "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
        ### EOF Data Preprocessing ###

        ### SOF Test Train Split ###
        if train_test_split:
            # take out a certain percentage of units from the training data set for testing later
            # (additionally to the classic validation methods)
            start = timer()
            units = data['serial_number'].unique()
            n_units = len(data['serial_number'].unique())

            units_test = random.sample(list(units), int(n_units * test_size))
            del n_units
            units_train = [nr for nr in units if nr not in units_test]
            del units

            df_n_test = data.loc[data['serial_number'].apply(lambda x: x in units_test)].copy()
            if df_n_test.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
            del units_test
            df_n_train = data.loc[data['serial_number'].apply(lambda x: x in units_train)].copy()
            if df_n_train.size==0: print("Model " + model + " skipped due to zero or one sample size."); continue
            del data
            del units_train

            df_n_test.to_pickle(data_path + "data_test_" + model + ".pkl")
            print("Test data set saved!" + "\n" +
                  "Serials: " + str(len(df_n_test['serial_number'].unique())) + ", Shape: " + str(df_n_test.shape))
            del df_n_test
            df_n_train.to_pickle(data_path + "data_train_" + model + ".pkl")
            print("Training data set saved!" + "\n" +
                  "Serials: " + str(len(df_n_train['serial_number'].unique())) + ", Shape: " + str(df_n_train.shape))
            end = timer()
            print("Task time: " + str(round(end - start, 2)) + "\n" +
            "Total time: " + str(round(end - start_total, 2)))
        elif svm | cox:
            start = timer()
            df_n_train = pd.read_pickle(data_path + "data_train_" + model + ".pkl")
            end = timer()
            print("Training data set loaded!" + "\n" +
                  "Serials: " + str(len(df_n_train['serial_number'].unique())) + ", Shape: " + str(df_n_train.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
        ### EOF Test Train Split ###

        ### SOF SVM ###
        if svm:
            # Do a simple support vector machine based regression based on all training data
            # define a helper function for the simple fitting through a support vector machine
            start = timer()
            X_train = df_n_train[list(set(df_n_train.columns.values) - set(cols_no_data))].values
            y_train = df_n_train['RUL'].values

            svm = sk.pipeline.Pipeline([
                ('scaler', sk.preprocessing.MinMaxScaler()),
                ('regression', sk.svm.SVR(gamma='scale', verbose=True)),
                ])

            #y_cv = sk.model_selection.cross_val_predict(
            #    svm,
            #    X_train,
            #    y_train,
            #    cv=k_cv,
            #    verbose=1,
            #)

            svm.fit(X_train, y_train)
            joblib.dump(svm, data_path + 'svm_model_' + model + '.pkl')

            # Load test set
            df_n_test = pd.read_pickle(data_path + "data_test_" + model + ".pkl")
            end = timer()
            print("SVM trained and test data set loaded!" + "\n" +
                  "Serials: " + str(len(df_n_test['serial_number'].unique())) + ", Shape: " + str(df_n_test.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            start = timer()
            X_test = df_n_test[list(set(df_n_test.columns.values) - set(cols_no_data))].values
            y_test = df_n_test['RUL'].values
            y_test_p = svm.predict(X_test)

            #print("cv test mse: %s" % scoring(y_cv, y_train))
            print("SVM  test " + scoring.__name__ + ": %s" % scoring(y_test_p, y_test))

            res_full = {
            #    'dy_train_cv': y_cv - y_train,
                'dy_test': y_test_p - y_test,
            #    'y_train_p_cv': y_cv,
                'y_test_p': y_test_p,
            }
            end = timer()
            print("SVM tested!"+ "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
        ### EOF SVM ###
        ### SOF Cox Regression ###
        if cox:
            start = timer()
            # read VIMP
            vimp = pd.read_csv(data_path + "vimp_" + model + ".csv")
            vimp_top = vimp.iloc[:,0].tolist()
            # vimp_top = vimp.iloc[0:29,0].tolist()
            # Create Correlation Matrix (Pearson)
            corr_matrix = df_n_train[vimp_top].corr().abs()
            #corr_matrix = df_n_train.drop(df_n_train[['Days_In_Service', 'failure']], axis=1).corr().abs()
            # Upper triangle
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # Find index of feature columns with correlation greater than n
            corr_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
            features = list(set(vimp_top) - set(corr_drop))
            features.extend(['failure', 'serial_number'])
            if not 'Days_In_Service' in features: features.extend(['Days_In_Service'])
            # Drop features
            xy_train = df_n_train[features].copy()
            # Drop columns if all constant
            cols_const = [col for col in xy_train.drop('failure', axis=1).columns if len(xy_train[col].unique()) <= 1]
            xy_train = xy_train.drop(xy_train[cols_const], axis=1).copy()
            end = timer()
            print("Highly correlated variables of Cox data set removed!" + "\n" +
                  "Shape: " + str(xy_train.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            # Using Cox Proportional Hazards model
            start = timer()
            cph = CoxPHFitter(penalizer=0.5)  ## Instantiate the class to create a cph object
            cph.fit(xy_train, 'Days_In_Service', event_col='failure', show_progress=True,
                    step_size=0.01, cluster_col='serial_number')
            joblib.dump(cph, data_path + 'cox_model_' + model + '.pkl')

            # Load test set
            df_n_test = pd.read_pickle(data_path + "data_test_" + model + ".pkl")
            end = timer()
            print("Cox trained and test data set loaded!" + "\n" +
                  "Serials: " + str(len(df_n_test['serial_number'].unique())) + ", Shape: " + str(df_n_test.shape) + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
            start = timer()
            # Filter test set for non data and highly correlated data
            #cols_no_data_cox = ['date', 'model', 'capacity_bytes', 'RUL', 'serial_number', 'failure']
            X_test = df_n_test[features].copy()
            xy_pred = pd.concat([pd.DataFrame(qth_survival_times(0.9, cph.predict_survival_function(X_test)).transpose()).reset_index(drop='index'),
                                 df_n_test.reset_index(drop='index')], axis=1)
            xy_pred['pred'] = xy_pred.iloc[:, 0] - xy_pred['Days_In_Service']
            #xy_pred = pd.concat([xy_pred, df_n_test.reset_index(drop='index')],  axis=1)
            # Remove infs and NaNs
            xy_pred = xy_pred[~xy_pred.isin([np.nan, np.inf, -np.inf]).any(1)]
            # Remove negative predictions
            xy_pred['pred'] *= (xy_pred['pred'] > 0)
            xy_pred.to_csv(data_path + "predictions_cox_" + model + ".csv")
            print("Cox test " + scoring.__name__ + ": %s" % scoring(xy_pred['pred'], xy_pred['RUL']))
            end = timer()
            print("Cox tested!" + "\n" +
                  "Task time: " + str(round(end - start, 2)) + "\n" +
                  "Total time: " + str(round(end - start_total, 2)))
    end = timer()
    print("Finished. Total time: " + str(round(end - start_total, 2)))