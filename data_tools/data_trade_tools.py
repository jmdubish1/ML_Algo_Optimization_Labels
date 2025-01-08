import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import data_tools.general_tools as gt
import sys
from data_tools.math_tools import find_percentile_for_percent_sum, clip_array

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler


class TradeData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.trade_data = self

        self.data_loc = str
        self.trade_df = pd.DataFrame()
        self.param_df = pd.DataFrame()
        self.analysis_df = pd.DataFrame()

        self.working_df = pd.DataFrame()
        self.y_train_df = pd.DataFrame()

        self.train_dates = []
        self.curr_test_date = None
        self.start_period_test_date = None
        self.test_dates = []
        self.y_test_df = pd.DataFrame()
        self.valid_params = []

        self.prep_trade_data()

    def prep_trade_data(self):
        self.get_trade_data()
        self.set_pnl()
        self.aggregate_labels()

    def get_trade_data(self):
        print('\nGetting Trade Data')
        self.trade_df = pd.read_feather(self.ph.pathmgr.trade_data_path())
        self.param_df = pd.read_feather(self.ph.pathmgr.trade_params_path())
        self.trade_df['DateTime'] = gt.adjust_datetime(self.trade_df['DateTime'])
        self.trade_df = (
            self.trade_df)[self.trade_df['DateTime'].dt.date >=
                           pd.to_datetime(self.ph.setup_params.start_train_date).date()]

    def set_dates(self, test_date):
        self.curr_test_date = pd.to_datetime(test_date)
        self.start_period_test_date = self.curr_test_date - timedelta(self.ph.setup_params.test_period_days)

    def set_pnl(self):
        self.trade_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                        self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                        self.trade_df['exitPrice'] - self.trade_df['entryPrice'])
        self.trade_df['PnL'] = self.trade_df['PnL']/self.trade_df['entryPrice'] * 100

    def aggregate_labels(self):
        self.trade_df['Label'] = self.trade_df[self.ph.setup_params.label_classes].idxmax(axis=1)
        self.trade_df.drop(columns=self.ph.setup_params.label_classes, inplace=True)

    def create_working_df(self):
        print('\nCreating Trades Work Df')
        self.working_df = self.trade_df[(self.trade_df['paramset_id'] == self.ph.paramset_id) &
                                        (self.trade_df['side'] == self.ph.side)]
        self.subset_start_time()

    def separate_train_test(self):
        print('\nSeparating Train-Test')
        self.subset_test_period()
        train_df = self.working_df[self.working_df['DateTime'] <= self.start_period_test_date]
        self.train_dates = list(np.unique(train_df['DateTime'].dt.date))
        self.y_train_df = train_df[['DateTime', 'Label']]

        test_df = self.working_df[(self.working_df['DateTime'] > self.start_period_test_date)]
        self.test_dates = list(np.unique(test_df['DateTime'].dt.date))
        self.y_test_df = test_df[['DateTime', 'Label']]

    def add_to_daily_dates(self, num_dates, train=True):
        add_days = []
        if train:
            initial_dates = self.train_dates[0]
        else:
            initial_dates = self.test_dates[0]

        for i in list(range(1, num_dates*2))[::-1]:
            add_days.append((initial_dates - timedelta(days=i)))

        return add_days

    def subset_test_period(self):
        self.working_df = self.working_df[(self.working_df['DateTime'].dt.date <= self.curr_test_date.date())]
        self.analysis_df = self.working_df.copy(deep=True)
        self.working_df = self.working_df[['DateTime', 'PnL', 'Label']]

        self.analysis_df['PnL'] = self.analysis_df['PnL'] * self.analysis_df['entryPrice'] / 100

    def subset_start_time(self):
        start_time = (
            pd.Timestamp(f'{self.ph.setup_params.start_hour:02}:{self.ph.setup_params.start_minute:02}:00').time())

        self.working_df['time'] = self.working_df['DateTime'].dt.time
        subset_df = self.working_df[
            (self.working_df['time'] >= start_time) &
            (self.working_df['time'] <= pd.Timestamp('16:00:00').time())]
        self.working_df = subset_df.drop(columns=['time'])










