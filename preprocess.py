import pandas as pd
import numpy as np


def extract_features(data_frame, column):
    data_frame[column] = pd.to_datetime(data_frame[column])

    data_frame['month'] = data_frame['booking_time'].dt.month
    data_frame['week'] = data_frame['booking_time'].dt.isocalendar().week
    data_frame['day'] = data_frame['booking_time'].dt.isocalendar().day
    
    return data_frame


def one_hot_coding(data_frame, columns):
    '''Creating Dummy Columns
    args:
        data_frame : data frame
        columns: list of column names
    return -- dataframe
        '''
    for column in columns:
        data_frame = pd.concat([data_frame, pd.get_dummies(data_frame[column],
                                                           prefix=column)], axis=1)
        data_frame.drop(column, inplace=True, axis=1)

    return data_frame


def normalize_columns(data_frame, columns):
    '''Normalizing columns
    args:
        data_frame : data frame
        columns: list of column names
    return -- dataframe
    '''
    for column in columns:
        mean = data_frame[column].mean()
        std = data_frame[column].std()
        if std != 0:
            data_frame[column] = (data_frame[column] - mean) / std
        else:
            data_frame[column] = 0.0
    return data_frame



def prepare_data(data_frame):
    '''
    Data Preparation to feed to the machine learning model
    '''
    data_frame = extract_features(data_frame, 'booking_time')
    drop_columns = ['payment_method_name', 'item_unique_cnt', 
                    'item_total_cnt', 'booking_time', 
                    'customer_id','merchant_id']
    data_frame = data_frame.drop(columns=drop_columns)
    data_frame = normalize_columns(data_frame, ['booking_distance', 'booking_amount'])
    data_frame = one_hot_coding(data_frame, ['day', 'week', 'month'])
    return data_frame


def impute_missing_values(data_frame, columns, is_numeric=False):
    '''
    Imputing Missing Values, Numeric with median, Categorical with Mode
    args:data_frame : data frame
        columns: list of column names
        is_numeric : Boolean
    return -- dataframe
    '''
    if is_numeric:
        logging.info("Imputing Missing Values for Numerical Columns")
    else:
        logging.info("Imputing Missing Values for Categorical Columns")
    for column in columns:
        if is_numeric:
            data_frame[column] = data_frame[column].fillna(data_frame[column].median())
        else:
            data_frame[column] = data_frame[column].fillna(data_frame[column].mode()[0])

    return data_frame