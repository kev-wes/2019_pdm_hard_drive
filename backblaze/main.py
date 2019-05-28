from backblaze.load_data import LoadData
from datetime import date
import pandas as pd
import random
from sklearn import pipeline
from sklearn.externals import joblib
import sklearn as sk
from scipy.stats import kurtosis
from sklearn import preprocessing
from sklearn import svm
import math

### SOF Parameters ###
data_path = "C:/Users/kwesendrup/PycharmProjects/master/backblaze/data/rawdata/"
cols_no_data = ['date', 'serial_number', 'model', 'capacity_bytes', 'failure', 'RUL']
# Set this to false to skip csv reading and load pickle if it was already saved
initialize_data_set = False
start_date = date(2016, 1, 1)
end_date = date(2017, 12, 31)
# Set this to false to skip preprocessing and load pickle if it was already saved
preprocess_data_set = True
filtered_models = ['ST12000NM0007']
# Reduce sample size. Choose value in ]0, 1]
sample_size = 1
# Remove hard drives that did not have measurements in the last n days. 0 does not remove any.
measurements_in_last_n_days = 0
# Removes measurements before degradation is occuring if True
elbow_point_detection = True
# Set this to false to skip model training and load pickle if it was already saved
train_test_split = True
test_size = 0.2
# Set this to false to skip model training and load pickle if it was already saved
train_and_pred_model = True
### EOF Parameters ###

### SOF Helper Functions ###
# Do a simple support vector machine based regression based on all training data
# define a helper function for the simple fitting through a support vector machine
def fit_sub(df_n_train, df_n_test, mdl, model):
    X_train = df_n_train[list(set(df_n_train.columns.values)- set(cols_no_data))].values
    y_train = df_n_train['RUL'].values

    X_test = df_n_test[list(set(df_n_test.columns.values)- set(cols_no_data))].values
    y_test = df_n_test['RUL'].values

    print(X_train.shape)
    if mdl is None:
        mdl = sk.pipeline.Pipeline([
            ('scaler', sk.preprocessing.MinMaxScaler()),
            ('regression', sk.svm.SVR(gamma='scale', verbose=True)),
        ])

    scoring = sk.metrics.mean_squared_error

    y_cv = sk.model_selection.cross_val_predict(
        mdl,
        X_train,
        y_train,
        cv=5,
        verbose=1,
    )

    mdl.fit(X_train, y_train)
    joblib.dump(mdl, data_path + 'model' + model + '.pkl')
    y_test_p = mdl.predict(X_test)

    scoring = sk.metrics.mean_squared_error

    print("cv test mse: %s" % scoring(y_cv, y_train))
    print("testing units mse: %s" % scoring(y_test_p, y_test))

    res_full = {
        'dy_train_cv': y_cv - y_train,
        'dy_test': y_test_p - y_test,
        'y_train_p_cv': y_cv,
        'y_test_p': y_test_p,
    }
    return res_full


def fit_and_test_clf(df_n_train, df_n_test, mdl=None, subspace=True, model=None):
    res_full = fit_sub(df_n_train, df_n_test, mdl, model)

    if not subspace:
        return res_full

    return res_full
### EOF Helper Functions ###

for model in filtered_models:
    print("STARTING MODEL " + model)
### SOF Initialize data set ###
    if initialize_data_set == True:
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
        print("Raw dataset initialized and loaded!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
    else:
        data = pd.read_pickle(data_path + "data_raw.pkl")
        print("Raw dataset loaded!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
    ### EOF Initialize data set ###

    ### SOF Data Preprocessing ###
    if preprocess_data_set == True:
        # Filter models
        print("Start of data preprocessing" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        data = data[data['model']==model].copy()
        print("After model reduction" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        # only 21 columns are not pure NaN or constant per model
        # Backblaze suggests only to use:
        # - "smart_5_raw": "Reallocated_Sector_Count",
        # - "smart_187_raw": "Reported_Uncorrectable_Errors",
        # - "smart_188_raw": "Command_Timeout",
        # - "smart_197_raw": "Current_Pending_Sector_Count",
        # - "smart_198_raw": "Offline_Uncorrectable"
        # Source: https://www.backblaze.com/blog/hard-drive-smart-stats/
        # -"smart_9_raw": "Days_In_Service" is also used

        data_pruned = data.rename(index=str, columns={"smart_5_raw": "Reallocated_Sector_Count",
                                                             "smart_9_raw": "Days_In_Service",
                                                             "smart_187_raw": "Reported_Uncorrectable_Errors",
                                                             "smart_188_raw": "Command_Timeout",
                                                             "smart_197_raw": "Current_Pending_Sector_Count",
                                                             "smart_198_raw": "Offline_Uncorrectable"}).copy()
        data_pruned = data_pruned[['date', 'serial_number', 'model', 'capacity_bytes', 'failure',
                            'Reallocated_Sector_Count', 'Days_In_Service', 'Reported_Uncorrectable_Errors',
                            'Command_Timeout', 'Current_Pending_Sector_Count', 'Offline_Uncorrectable']].copy()
        # TODO Remove normalized values by Regex?
        # data_pruned = data_pruned[['date', 'serial_number', 'model', 'capacity_bytes', 'failure',
        #                    'Reallocated_Sector_Count', 'Days_In_Service', 'Reported_Uncorrectable_Errors',
        #                    'Command_Timeout', 'Current_Pending_Sector_Count', 'Offline_Uncorrectable']].copy()
        data_pruned = data_pruned.sort_values(['serial_number', 'date'], ascending=[True, False])
        # Assign RUL by counting upwards per serial number in the descended data frame
        data_pruned['RUL'] = data_pruned.groupby((data_pruned['serial_number'] !=
                                                  data_pruned['serial_number'].shift(1)).cumsum()).cumcount()
        data_pruned = data_pruned.sort_values(['serial_number', 'date'], ascending=True)
        # Divide hours in service by 24 to get days of service
        data_pruned[['Days_In_Service']] = data_pruned[['Days_In_Service']].div(24).round(0)
        # Remove entries where RUL = 0 and Days in Service = 0
        data_pruned = data_pruned[-((data_pruned.Days_In_Service == 0) &
                                    (data_pruned.RUL == 0))]
        print("After pruning features and adding RUL" + "\n" +
              "Serials: " + str(len(data_pruned['serial_number'].unique())) + ", Shape: " + str(data_pruned.shape))
        # Remove hard drives that have no measurements in last 10 days
        if measurements_in_last_n_days != 0:
            df_lt_10 = data_pruned.loc[data_pruned['RUL'] <= measurements_in_last_n_days].copy()
            df_lt_10 = df_lt_10.loc[(df_lt_10['Reallocated_Sector_Count'] != 0) |
                                 (df_lt_10['Reported_Uncorrectable_Errors'] != 0) |
                                 (df_lt_10['Command_Timeout'] != 0) |
                                 (df_lt_10['Current_Pending_Sector_Count'] != 0) |
                                 (df_lt_10['Offline_Uncorrectable'] != 0)].copy()
            serials_w_data = df_lt_10['serial_number'].unique().tolist()
            data_pruned = data_pruned[data_pruned['serial_number'].isin(serials_w_data)].copy()
            print("After pruning hard drives w/o non-zero values in last " + str(measurements_in_last_n_days) + " days" + "\n" +
                 "Serials: " + str(len(data_pruned['serial_number'].unique())) + ", Shape: " + str(data_pruned.shape))

        # Min Max Normalization
        # TODO Change to Mean Var Normalization?
        data_norm = data_pruned.copy()
        # Loop only over data columns
        for feature_name in list(set(data_norm.columns.values)- set(cols_no_data)):
            max_value = data_pruned[feature_name].max()
            min_value = data_pruned[feature_name].min()
            data_norm[feature_name] = (data_pruned[feature_name] - min_value) / (max_value - min_value)
        # Free space
        del data_pruned

        # Drop NaN values
        print("NaNs:\n" + str(data_norm.isnull().sum(axis = 0)))
        # Drop columns if all NaN

        data = data_norm.dropna(axis=1, how='all')
        print("After dropping NaN columns" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        # Drop rows if any NaN
        # TODO Mean values of previous and next timestamp
        print("NaNs:\n" + str(data.isnull().sum(axis=0)))
        data = data.dropna()
        print("After dropping NaN rows" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))

        # Feature Engineering
        # Zhao, R., Yan, R., Wang, J., and Mao, K. 2017. “Learning to Monitor Machine Health with
        # Convolutional Bi-Directional LSTM Networks,” Sensors, (17:2), p. 273.
        for feature_name in list(set(data_norm.columns.values) - set(cols_no_data)):
            # Mean
            data[feature_name + '_mean'] = data.groupby(['serial_number']).apply(
                lambda x: x[feature_name].rolling(5).mean()).reset_index(level=0,drop=True)
             # Min
            data[feature_name + '_min'] = data.groupby(['serial_number']).apply(
                lambda x: x[feature_name].rolling(5).min()).reset_index(level=0, drop=True)
            # Max
            data[feature_name + '_max'] = data.groupby(['serial_number']).apply(
                lambda x: x[feature_name].rolling(5).max()).reset_index(level=0, drop=True)
            # Variance
            data[feature_name + '_var'] = data.groupby(['serial_number']).apply(
                lambda x: x[feature_name].rolling(5).var()).reset_index(level=0, drop=True)
            # Root mean square
            data[feature_name + '_rms'] = data.groupby(['serial_number']).apply(
                lambda x: math.sqrt(1/5*(x[feature_name].rolling(5))**2)).reset_index(level=0, drop=True)
        print("After feature engineering" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        # Drop NaNs after sliding window
        data = data.dropna()
        print("After dropping NaN through sliding window of feature engineering rows" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        # Reduce sample size
        serials = data['serial_number'].unique()
        len_serials = len(serials)
        serials_filter = random.sample(list(serials), int(len_serials * sample_size))
        data = data.loc[data['serial_number'].apply(lambda x: x in serials_filter)].copy()
        print("After sample size reduction" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        # Remove rows before first non-zero measurement
        # Some kind of elbow point detection
        if elbow_point_detection == True:
            serials = data['serial_number'].unique()
            df2 = pd.DataFrame()
            df = pd.DataFrame()
            for ser in serials:
                df = data[data['serial_number'] == ser].copy()
                df = df[((df.Reallocated_Sector_Count != 0).cumsum() > 0) |
                        ((df.Reported_Uncorrectable_Errors != 0).cumsum() > 0) |
                        ((df.Command_Timeout != 0).cumsum() > 0) |
                        ((df.Current_Pending_Sector_Count != 0).cumsum() > 0) |
                        ((df.Offline_Uncorrectable != 0).cumsum() > 0)]
                df2 = df2.append(df)
            del df
            data = df2.copy()
            del df2
            print("After removal of rows before elbow point" + "\n" +
                    "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
        data.to_pickle(data_path + "data_preprocessed.pkl")
        print("Dataset preprocessed and saved!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
    else:
        data = pd.read_pickle(data_path + "data_preprocessed.pkl")
        print("Preprocessed dataset loaded!" + "\n" +
              "Serials: " + str(len(data['serial_number'].unique())) + ", Shape: " + str(data.shape))
    ### EOF Data Preprocessing ###

    ### SOF Test Train Split ###
    if train_test_split == True:
        # take out a certain percentage of units from the training data set for testing later
        # (additionally to the classic validation methods)
        units = data['serial_number'].unique()
        n_units = len(data['serial_number'].unique())

        units_test = random.sample(list(units), int(n_units * test_size))
        units_train = [nr for nr in units if nr not in units_test]

        df_n_test = data.loc[data['serial_number'].apply(lambda x: x in units_test)].copy()
        df_n_train = data.loc[data['serial_number'].apply(lambda x: x in units_train)].copy()

        df_n_test.to_pickle(data_path + "data_test" + model + ".pkl")
        df_n_train.to_pickle(data_path + "data_train" + model + ".pkl")
        print("Training data set saved!" + "\n" +
              "Serials: " + str(len(df_n_train['serial_number'].unique())) + ", Shape: " + str(df_n_train.shape))
        print("Test data set saved!" + "\n" +
              "Serials: " + str(len(df_n_test['serial_number'].unique())) + ", Shape: " + str(df_n_test.shape))
    else:
        df_n_test = pd.read_pickle(data_path + "data_test" + model + ".pkl")
        df_n_train = pd.read_pickle(data_path + "data_train" + model + ".pkl")
        print("Training data set loaded!" + "\n" +
              "Serials: " + str(len(df_n_train['serial_number'].unique())) + ", Shape: " + str(df_n_train.shape))
        print("Test data set loaded!" + "\n" +
              "Serials: " + str(len(df_n_test['serial_number'].unique())) + ", Shape: " + str(df_n_test.shape))
    ### EOF Test Train Split ###

    ### SOF Model Training and Prediction ###

    if train_and_pred_model == True:
        res_full = fit_and_test_clf(df_n_train, df_n_test, mdl=None, model=model)
        print("Model trained!")
    else:
        model = joblib.load(data_path + 'model' + model + '.pkl')
        print("Trained model loaded!")
    ### EOF Model Training and Prediction ###

print("Finished.")