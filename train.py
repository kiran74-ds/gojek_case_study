# pylint: disable=missing-module-docstring,import-error


import pickle
from logging.config import dictConfig
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from preprocess import prepare_data
import os


def prepare_data_and_train_model():
    '''
    Prepare and train the ML model on input data
    :return: model and test data
    '''
    data_frame = pd.read_csv("data/Assignment_train.csv")
    data_frame = prepare_data(data_frame)
    
    train_set, validation_set = train_test_split(data_frame, random_state=42, test_size=0.2)
    predictors = train_set.columns[~train_set.columns.isin(['actual_total_time'])]
    
    rf_reg = RandomForestRegressor(n_estimators=10,
                               max_depth=5,
                               min_samples_leaf=2,
                               max_features=0.7)
    rf_reg.fit(train_set[predictors], train_set['actual_total_time'])

    yhat = rf_reg.predict(train_set[predictors])

    pickle.dump(rf_reg, open('model_weights', 'wb'))
  
    return rf_reg


def get_predictions(test_data_path):
    '''
    make predictions on the test data
    :return: predictions
    '''
    test_data = pd.read_csv(test_data_path, parse_dates = True)
    loaded_model = pickle.load(open('model_weights', 'rb'))
    X_test = prepare_data(test_data)
    predictors = X_test.columns[~X_test.columns.isin(['actual_total_time'])]
    predictions = loaded_model.predict(X_test[predictors])
    test_data['predictions'] = predictions
    
    return predictions


if __name__ == "__main__":
    model  = prepare_data_and_train_model()
    test_predictions = get_predictions('data/Assignment_train.csv')
    print(test_predictions)
