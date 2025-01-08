import numpy as np
import pandas as pd
from datetime import timedelta


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