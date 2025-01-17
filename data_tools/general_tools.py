import numpy as np
import pandas as pd
from datetime import timedelta
import hashlib


def ensure_friday(date):
    weekday = date.weekday()

    if weekday != 4:
        days_until_test_date = (4 - weekday) % 7
        date = date + timedelta(days=days_until_test_date)

    return date


def adjust_datetime(datetimes):
    datetimes = pd.to_datetime(datetimes, format='%Y-%m-%d %H:%M:%S')
    return datetimes


def convert_date_to_dt(data):
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'],
                                      format='%m/%d/%Y %H:%M')
    data.drop(columns=['Date', 'Time'], inplace=True)

    return data


def set_month_day(df, time_frame):
    df['Month'] = pd.to_datetime(df['DateTime']).dt.month
    df['Day'] = pd.to_datetime(df['DateTime']).dt.day
    df['DayofWeek'] = pd.to_datetime(df['DateTime']).dt.dayofweek

    if time_frame != 'daily':
        df['Hour'] = pd.to_datetime(df['DateTime']).dt.hour
        df['Minute'] = pd.to_datetime(df['DateTime']).dt.minute

    return df


def merge_dfs(df_list):
    first_df = df_list[0]
    for df in df_list[1:]:
        first_df = pd.merge(first_df, df, on='DateTime')

    return first_df


def fill_na_inf(df):
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def pad_to_length(arr, length, pad_value=0):
    if arr.shape[0] >= length:
        return arr[-length:]
    padding = np.full((length - arr.shape[0], arr.shape[1]), pad_value)
    return np.vstack((padding, arr))


def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def generate_unique_id(data_dict, paramset_id):
    """
    Generate a unique ID based on a dictionary containing `paramset_id` and XGBoost params.

    Parameters:
        data_dict (dict): A dictionary with keys such as `paramset_id`, `max_depth`, etc.
        paramset_id: the ID of the Algo Paramset
    Returns:
        str: A unique ID.
    """

    if paramset_id is None:
        paramset_id = data_dict['paramset_id']

    columns = ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'n_estimators']
    if not isinstance(data_dict, dict):
        data_dict = {col: data_dict[col] for col in columns}

    unique_string = "_".join(str(data_dict[col]) for col in columns)

    hashed_id = hashlib.md5(unique_string.encode()).hexdigest()[:12]
    full_id = f"Algo_{paramset_id}_{hashed_id}"

    return full_id