import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import warnings
import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler

warnings.simplefilter(action='ignore', category=FutureWarning)


class MLTrainData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.ml_data = self

        self.x_train_daily = None
        self.x_test_daily = None

        self.x_train_intra = None
        self.x_test_intra = None

        self.y_train_df = None
        self.y_test_df = None
        self.xy_train_intra = None
        self.xy_test_intra = None

        self.intra_scaler = None
        self.daily_scaler = None
        self.y_pnl_scaler = None
        self.y_wl_onehot_scaler = OneHotEncoder(sparse_output=False)
        self.y_wl_label_encoder = LabelEncoder()

    def prep_train_test_data(self, load_scalers):
        self.set_x_train_test_datasets()
        self.scale_x_data(load_scalers)
        self.onehot_y_wl_data()

        self.merge_x_y()

    def set_x_train_test_datasets(self):
        print('\nBuilding X-Train and Test Datasets')
        mktw = self.ph.mktdata_working
        self.x_train_intra = (
            mktw.intra_working)[mktw.intra_working['DateTime'].dt.date.isin(self.ph.trade_data.train_dates)]

        self.x_test_intra = (
            mktw.intra_working)[mktw.intra_working['DateTime'].dt.date.isin(self.ph.trade_data.test_dates)]

        train_dates = (
                self.ph.trade_data.add_to_daily_dates(self.ph.ml_model.daily_len, train=True) +
                self.ph.trade_data.train_dates)
        self.x_train_daily = mktw.daily_working[mktw.daily_working['DateTime'].dt.date.isin(train_dates)]
        self.x_train_daily.reset_index(inplace=True, drop=True)

        test_dates = (
                self.ph.trade_data.add_to_daily_dates(self.ph.ml_model.daily_len, train=False) +
                self.ph.trade_data.test_dates)
        self.x_test_daily = mktw.daily_working[mktw.daily_working['DateTime'].dt.date.isin(test_dates)]
        self.x_test_daily.reset_index(inplace=True, drop=True)

    def scale_x_data(self, load_scalers=False):
        print('\nScaling X Data')
        self.x_train_intra.iloc[:, 1:] = self.x_train_intra.iloc[:, 1:].astype('float32')

        self.x_test_intra.iloc[:, 1:] = self.x_test_intra.iloc[:, 1:].astype('float32')
        self.x_train_daily.iloc[:, 1:] = self.x_train_daily.iloc[:, 1:].astype('float32')
        self.x_test_daily.iloc[:, 1:] = self.x_test_daily.iloc[:, 1:].astype('float32')

        if self.ph.train_modeltf:
            if load_scalers:
                print('Using Previously Loaded X-Scalers')

            else:
                print('Creating New X-Scalers')
                self.intra_scaler = StandardScaler()
                self.intra_scaler.fit(self.x_train_intra.iloc[:, 1:].values)

                self.daily_scaler = StandardScaler()
                self.daily_scaler.fit(self.x_train_daily.iloc[:, 1:].values)

        self.x_train_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_train_intra.iloc[:, 1:].values)
        self.x_test_intra.iloc[:, 1:] = self.intra_scaler.transform(self.x_test_intra.iloc[:, 1:].values)

        self.x_train_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_train_daily.iloc[:, 1:].values)
        self.x_test_daily.iloc[:, 1:] = self.daily_scaler.transform(self.x_test_daily.iloc[:, 1:].values)

    def onehot_y_wl_data(self):
        print('\nOnehotting WL Data')
        resample_tf = self.ph.setup_params.over_sample

        def encode_dataframe(df, scaler):
            if len(df) == 0:
                return df

            labels = df.iloc[:, 1].values.reshape(-1, 1)
            enc_labels = scaler.transform(labels)
            label_names = scaler.get_feature_names_out()

            for name, ind in zip(label_names, range(len(label_names))):
                df[name] = enc_labels[:, ind]

            return df

        self.y_train_df = self.ph.trade_data.y_train_df.copy(deep=True)
        if resample_tf:
            print(f'Trades before Resample: {len(self.y_train_df)}')
            self.y_train_df = balance_y_trades(self.y_train_df)
            print(f'Trades after Resample: {len(self.y_train_df)}')

        train_labels = self.y_train_df.iloc[:, 1].values.reshape(-1, 1)
        wl_dat = self.y_wl_onehot_scaler.fit_transform(train_labels)
        label_names = self.y_wl_onehot_scaler.get_feature_names_out()

        for name, ind in zip(label_names, range(len(label_names))):
            self.y_train_df[name[3:]] = wl_dat[:, ind] #seems wrong

        # Encode test data using the fitted scaler
        self.y_test_df = self.ph.trade_data.y_test_df.copy(deep=True)
        self.y_test_df = encode_dataframe(self.y_test_df, self.y_wl_onehot_scaler)

    def merge_x_y(self):
        ncols = self.ph.setup_params.num_y_cols
        self.xy_train_intra =\
            pd.merge(self.x_train_intra, self.y_train_df, on='DateTime', how='left').reset_index(drop=True)
        self.xy_train_intra.iloc[:, -ncols:] = self.xy_train_intra.iloc[:, -ncols:].shift()
        self.xy_train_intra = self.xy_train_intra.fillna(0)
        self.xy_train_intra.drop(columns='Label', inplace=True)
        self.xy_train_intra.reset_index(inplace=True, drop=True)

        self.xy_test_intra =\
            pd.merge(self.x_test_intra, self.y_test_df, on='DateTime', how='left').reset_index(drop=True)
        self.xy_test_intra.iloc[:, -ncols:] = self.xy_test_intra.iloc[:, -ncols:].shift()
        self.xy_test_intra = self.xy_test_intra.fillna(0)
        self.xy_test_intra.drop(columns='Label', inplace=True)
        self.xy_test_intra.reset_index(inplace=True, drop=True)


def balance_y_trades(y_df):
    """
    Balance y_df that contains both string labels and timestamps.

    Parameters:
    - y_df (pd.DataFrame): DataFrame with 'Label' (string) and 'Timestamp'.

    Returns:
    - Resampled y_df with balanced classes and aligned timestamps.
    """
    y_labels = y_df['Label'].astype(str)
    y_with_index = pd.DataFrame({'index': y_df.index, 'Label': y_labels})

    oversampler = RandomOverSampler()
    y_with_index, y_labels_resampled = oversampler.fit_resample(y_with_index, y_labels)

    resampled_indices = y_with_index['index']
    y_timestamps_resampled = y_df.loc[resampled_indices, 'DateTime'].values

    resampled_df = pd.DataFrame({
        'DateTime': y_timestamps_resampled,
        'Label': y_labels_resampled
    })
    resampled_df = resampled_df.sort_values(by='DateTime', ascending=True).reset_index(drop=True)

    return resampled_df

