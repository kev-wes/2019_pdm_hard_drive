from datetime import timedelta
import pandas as pd

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)+1):
        yield start_date + timedelta(n)

def LoadData(path, start_date, end_date, failure = None, serial = None):
    data = pd.DataFrame()
    for single_date in daterange(start_date, end_date):
        data = data.append(pd.read_csv(path + single_date.strftime("%Y-%m-%d") + '.csv'))
        if failure is not None:
            data = data.loc[data['failure'] == failure]
            print('Entries of ' + single_date.strftime("%Y-%m-%d") + '.csv with failure == ' + str(failure) + ' loaded!')
        if serial is not None:
            data = data.loc[data['serial_number'].isin(serial)]
            print('Entries of ' + single_date.strftime("%Y-%m-%d") + '.csv with failed drive serial number loaded!')



    return data