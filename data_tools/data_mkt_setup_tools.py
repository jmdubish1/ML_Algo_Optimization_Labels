import pandas as pd
import numpy as np
import data_tools.general_tools as gt
import data_tools.math_tools as mt
import warnings
from fracdiff.sklearn import Fracdiff

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler

pd.set_option('display.max_columns', 500)

warnings.simplefilter(action='ignore', category=FutureWarning)


class MktDataSetup:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.mkt_setup = self
        self.all_secs = [self.ph.setup_params.security] + self.ph.setup_params.other_securities

        self.dailydata_clean = pd.DataFrame()
        self.intradata_clean = pd.DataFrame()
        self.security_df = pd.DataFrame()

        self.load_prep_data('daily')
        self.load_prep_data('intraday')

    def load_prep_data(self, time_frame):
        if time_frame == 'daily':
            data_end = 'daily_20240505_20040401.txt'
        else:
            data_end = f'{self.ph.pathmgr.time_frame_train}_20240505_20040401.txt'
        print(f'\nLoading {time_frame} data')

        dfs = []
        for sec in self.all_secs:
            print(f'...{sec}')
            temp_df = pd.read_csv(f'{self.ph.pathmgr.data_loc}\\{sec}_{data_end}')
            temp_df = gt.convert_date_to_dt(temp_df)

            temp_df = temp_df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Vol']]

            if sec == self.ph.setup_params.security:
                self.security_df = temp_df.copy(deep=True)

            for col in temp_df.columns[1:]:
                temp_df.rename(columns={col: f'{sec}_{col}'}, inplace=True)

            dfs.append(temp_df)

        if time_frame == 'daily':
            self.dailydata_clean = gt.merge_dfs(dfs)
            self.dailydata_clean = gt.set_month_day(self.dailydata_clean, time_frame)
        else:
            self.intradata_clean = gt.merge_dfs(dfs)
            self.intradata_clean = gt.set_month_day(self.intradata_clean, time_frame)


class MktDataWorking:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.mktdata_working = self
        self.daily_working = pd.DataFrame()
        self.intra_working = pd.DataFrame()
        self.ffd_df = pd.read_excel(f'{self.ph.pathmgr.trade_dat_loc}\\agg_data\\all_FFD_params.xlsx')

        self.param_id_df = pd.DataFrame()
        self.fastema = 12

        self.prep_working_data('daily')
        self.prep_working_data('intraday')
        self.subset_start_time()

    def prep_working_data(self, time_frame):
        print(f'Prepping Working Data: {time_frame} ')
        if time_frame == 'daily':
            df = self.ph.mkt_setup.dailydata_clean.copy(deep=True)
            self.fastema = 8
        else:
            df = self.ph.mkt_setup.intradata_clean.copy(deep=True)

        df = mt.sec_correlation(df, self.ph.mkt_setup.all_secs)

        for sec in self.ph.mkt_setup.all_secs:
            if time_frame != 'daily':
                df_daily_temp = self.daily_working.copy(deep=True)
                df = diff_between_prev_moving_avg(df, df_daily_temp, sec)
            df[f'{sec}_ATR_fast'] = build_atr(df, sec, 6)
            df[f'{sec}_ATR_slow'] = build_atr(df, sec, 10)
            df[f'{sec}_RSI_k'], df[f'{sec}_RSI_d'] = mt.create_smooth_rsi(df[f'{sec}_Close'], self.fastema)

            df = self.frac_diff(df, sec)
            df = prep_ema(df, sec, [100, 4, self.fastema, 32])
            df = mt.add_high_low_diff(df, sec)
            df[f'{sec}_Upper_BB'], df[f'{sec}_Lower_BB'] = (
                mt.calculate_bollinger_bands(df, sec, length=self.fastema, std_dev=2.0))
            df = mt.garch_modeling(df, sec)

        df = mt.encode_time_features(df, time_frame)
        df = gt.fill_na_inf(df)

        if time_frame == 'daily':
            self.daily_working = df
        else:
            self.intra_working = df

    def frac_diff(self, df, sec):
        print(f'{sec}...Fractionally Differentiating')
        for met in ['Close', 'Vol']:
            d_val, ws = self.get_frac_diff_params(met, sec)
            ws = min(ws, 90)

            if met == 'Close':
                fracdiff_model = Fracdiff(d=d_val, window=ws)
                fracdiff_model.fit(df[f'{sec}_Close'])

                diff_close = fracdiff_model.transform(df[f'{sec}_Close'].to_frame()).squeeze()
                df[f'{sec}_Open'] = (df[f'{sec}_Open'] / df[f'{sec}_Close']) * diff_close
                df[f'{sec}_High'] = (df[f'{sec}_High'] / df[f'{sec}_Close']) * diff_close
                df[f'{sec}_Low'] = (df[f'{sec}_Low'] / df[f'{sec}_Close']) * diff_close
                df[f'{sec}_Close'] = diff_close

            else:
                fracdiff_model = Fracdiff(d=d_val, window=ws)
                df[f'{sec}_{met}'] = fracdiff_model.fit_transform(df[f'{sec}_Vol'].to_frame())

        return df

    def get_frac_diff_params(self, met, sec):
        met = 'Close' if met in ['Open', 'High', 'Low'] else met
        df = self.ffd_df.loc[(self.ffd_df['time_frame'] == self.ph.setup_params.time_frame_train) &
                             (self.ffd_df['security'] == sec) &
                             (self.ffd_df['Data'] == met)].reset_index(drop=True)

        d_val, ws = df.loc[0, 'd_val'], df.loc[0, 'window']

        return d_val, ws

    def subset_start_time(self):
        start_time = (
            pd.Timestamp(f'{self.ph.setup_params.start_hour:02}:{self.ph.setup_params.start_minute:02}:00').time())

        self.intra_working['time'] = self.intra_working['DateTime'].dt.time
        subset_df = self.intra_working[
            (self.intra_working['time'] >= start_time) &
            (self.intra_working['time'] <= pd.Timestamp('16:00:00').time())]
        self.intra_working = subset_df.drop(columns=['time'])

        self.intra_working['time'] = self.intra_working['DateTime'].dt.time
        subset_df = self.intra_working[
            (self.intra_working['time'] >= start_time) &
            (self.intra_working['time'] <= pd.Timestamp('16:00:00').time())]
        self.intra_working = subset_df.drop(columns=['time'])


def prep_ema(df, sec, ema_lens):
    max_ema = np.max(ema_lens)
    for ema_len in ema_lens:
        df[f'{sec}_EMA_{ema_len}'] = mt.calculate_ema_numba(df, f'{sec}_Close', ema_len)
        df[f'{sec}_EMA_Close_{ema_len}'] = (df[f'{sec}_Close'] - df[f'{sec}_EMA_{ema_len}']) / df[f'{sec}_Close']
        if ema_len != max_ema:
            df[f'{sec}_EMA_{ema_len}'] = df[f'{sec}_EMA_{ema_len}'] / df[f'{sec}_EMA_{max_ema}']

    return df


def build_atr(df, sec, speed):
    atr_arr = mt.create_atr(df, sec, n=speed)
    atr_arr = np.nan_to_num(atr_arr, 1)
    ema = mt.calculate_ema_numba(df, f'{sec}_Close', speed)
    atr_arr = atr_arr/ema

    return atr_arr


def diff_between_prev_moving_avg(df_intra, df_daily, sec):
    df_intra['Date'] = df_intra['DateTime'].dt.date.shift()
    df_daily['Date'] = df_daily['DateTime'].dt.date
    merged_data = pd.merge(
        df_intra,
        df_daily[['Date', f'{sec}_EMA_Close_100']],
        on='Date',
        how='left'
    )

    merged_data[f'{sec}_Close_Daily_diff'] = (
            (merged_data[f'{sec}_Close'] - merged_data[f'{sec}_EMA_Close_100']) / merged_data[f'{sec}_EMA_Close_100'])

    merged_data.drop(columns='Date', inplace=True)

    return merged_data

