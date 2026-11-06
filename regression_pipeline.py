#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by lawa_ap | 24.10.2025


:Authors: Apurva Lawate

:Version: 0.1.0

:Project: point_machine

Summary
-------
{Description: Summary}

Usage
-----
{Description: Usage}

Changes
-------
* 24.10.2024 - Version 0.1.0: setup

"""

__version__ = '0.1.0'  # Has to be consistent with header!

# Built-in/Generic Imports (os, sys, ...)
import os
import datetime

# Common Libs (numpy, pandas, ...)

import sqlalchemy
import pandas as pd
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Own modules
import src.config as config
import src.visualization.utils as vutils


# data acquisition
def data_acquisition(timestamp_start: datetime.datetime, timestamp_end: datetime.datetime,
                     active_motor_current_sensor_count_: int, columns: list[str]) -> pd.DataFrame:

    """
    :param timestamp_start: lower threshold for the data
    :param timestamp_end: upper threshold for the data
    :param active_motor_current_sensor_count_: Desired number of active motor current sensors in the same time
    :param columns: columns to consider (timestamp is always added)
    :return: synchronized data
    """
    engine = sqlalchemy.create_engine(url=config.DB_URL_SQA)

    columns_str = ', '.join(list(set(columns + ['timestamp'])))

    query = f"""SELECT {columns_str}
                FROM {schema}.ps_mc_time_synced AS psmc
                WHERE psmc.active_motor_current_sensor_count > {active_motor_current_sensor_count_}
                AND (psmc.timestamp >= %(timestamp_start)s AND psmc.timestamp <= %(timestamp_end)s)
                ;"""

    params = {'timestamp_start': timestamp_start,
              'timestamp_end': timestamp_end
              }

    with engine.connect() as conn:
        data = pd.read_sql_query(query, conn, params=params)

    data['timestamp_unix'] = data['timestamp'].apply(lambda x: x.timestamp())

    data.dropna(subset=[col for col in data.columns if not (col.startswith('motor_current') or col == 'cumulative_motor_current')], inplace=True)
    data.fillna(value=0, inplace=True)

    return data


# Defining features and target variables
def feature_target_selection_and_split(data: pd.DataFrame, features_variation: list, target_: str)\
        -> [pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    :param data: Entire dataset
    :param features_variation: list of desired features
    :param target_: string with the column name of target variable
    :return: split data for training and testing
    """

    X = data[features_variation]  # features
    y = data[target_]  # target

    # if 'timestamp' in features_variation:
    #     new = X['timestamp'].apply(lambda x: x.timestamp())
    #     X = X.drop(columns=['timestamp'])
    #     X['timestamp'] = new

    if len(features_variation) == 1:  # Sklearn needs 2D features, in case of pandas series, we have to reshape it to 2D
        data.values.reshape(-1, 1)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.25, random_state=42)

    return X_train_, X_test_, y_train_, y_test_


# Model selection and parameters initialization
def modeling(model_name, **kwargs):

    model_ = model_name(**kwargs)
    return model_


def model_evaluation(true_values: pd.Series, predicted_values: pd.Series) -> [float, float]:
    """
    :param true_values: True target values from the test data
    :param predicted_values: Predictions on the test data
    :return: Evaluation metrics - mean squared error and R2 score
    """
    mse_ = mean_squared_error(true_values, predicted_values)
    r2_ = r2_score(true_values, predicted_values)

    return mse_, r2_


def plots(time_start: datetime.datetime, time_end: datetime.datetime, save_loc: str,
          plot_stats_: dict, *args):
    """
    :param time_start: lower threshold for the data
    :param time_end: upper threshold for the data
    :param save_loc: Location to save the plots
    :param plot_stats_: a dictionary consisting of key information on the model:
    Name of model and evaluation metric scores
    :param args: univariate data for histogram
    :return: nothing
    """
    # plotting the histogram
    if len(args) == 1:
        plot_data = args[0]

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.hist(plot_data, bins=30)

        ax.set_title(f'Residual plot for duration {time_end - time_start} from {time_start} to {time_end}')
        ax.set_xlabel('Actual - Predicted')
        ax.set_ylabel('Frequency')

        ax.text(0.7, 0.95, f"r2 score = {plot_stats_['r2']}\nmean squared error = {plot_stats_['mse']}",
                transform=ax.transAxes, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))

        # Saving the plot
        plt.savefig(os.path.join(save_loc,
                                 f"histogram_{plot_stats_['model_name']}_{time_start.strftime('%Y%m%dT%H%M%SZ')}_to_"
                                 f"{time_end.strftime('%Y%m%dT%H%M%SZ')}.png"))

        plt.close()
    else:
        print("Please pass univariate data")


if __name__ == '__main__':
    schema = 'wayside_processed_akima'

    save_to = os.path.join(config.folderpath_plots, f'regression/{schema}')
    os.makedirs(save_to, exist_ok=True)

    # INPUTS

    # 1. SQL Query
    time_begin = datetime.datetime(year=2024, month=2, day=26, tzinfo=datetime.timezone.utc)
    time_ending = datetime.datetime(year=2024, month=8, day=16, tzinfo=datetime.timezone.utc)
    active_motor_current_sensor_count = 0

    # 2. Data - Features and Target
    # 'power_supply_baseline', 'cumulative_motor_current', 'power_supply'
    # 'motor_current_4735',  'motor_current_56523', 'motor_current_56524', 'motor_current_56525', 'motor_current_56526'
    features = ['power_supply_baseline', 'cumulative_motor_current']
    target = 'power_supply'
    add_columns = []  # temperature_c group

    # 3. Model
    model = LinearRegression  # Available models - LinearRegression, Ridge, PoissonRegressor, GammaRegressor
    parameters = {'fit_intercept': False}
    # parameters = {'alpha': 1}

    # ROUTINE

    # Step 1: Data acquisition and splitting
    synced_data = data_acquisition(time_begin, time_ending, active_motor_current_sensor_count,
                                   features + [target] + add_columns)
    X_train, X_test, y_train, y_test = feature_target_selection_and_split(synced_data, features, target)

    # Step 2: Fit and Predict with the model
    pipeline = Pipeline([('model', modeling(model, **parameters))])

    pipeline.fit(X_train, y_train)
    y_predicted = pipeline.predict(X_test)

    # Step 3: Metrics based evaluation
    mse, r2 = model_evaluation(y_test, pd.Series(y_predicted))
    if model == Ridge or model == LinearRegression:
        print(pipeline['model'].coef_)
    if model == LinearRegression:
        print(pipeline['model'].intercept_)

    # Step 4: Preprocessing and plotting for histogram and the regression plot
    residuals = y_test - y_predicted
    residuals = residuals.to_numpy()
    filtered_for_hist = residuals[~vutils.is_outlier(residuals)]

    plot_stats = {'model_name': model.__name__, 'mse': round(mse, 3), 'r2': round(r2, 3)}
    plots(time_begin, time_ending, save_to, plot_stats, filtered_for_hist)
    print('saved the histogram')

    print()
