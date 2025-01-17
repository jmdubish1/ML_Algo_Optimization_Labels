import numpy as np
import pandas as pd
from datetime import timedelta
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler


class AlgoParamResults:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.param_chooser = self
        self.end_date = pd.to_datetime(self.ph.setup_params.final_test_date, format='%Y-%m-%d')
        self.trade_folder = self.ph.pathmgr.algo_dat_loc
        self.trades_file = self.ph.pathmgr.trade_data_path()
        self.params_file = self.ph.pathmgr.trade_params_path()

        self.trade_df = pd.DataFrame
        self.param_df = pd.DataFrame
        self.pnl_df = pd.DataFrame
        self.best_params_df = pd.DataFrame
        self.valid_params = np.array
        self.given_paramsets = self.ph.setup_params.chosen_params

    def load_files(self):
        self.trade_df = pd.read_feather(self.trades_file)
        self.param_df = pd.read_feather(self.params_file)

    def set_pnl(self):
        self.pnl_df = self.trade_df.copy(deep=True)
        self.pnl_df['PnL'] = np.where(self.trade_df['side'] == 'Bear',
                                      self.trade_df['entryPrice'] - self.trade_df['exitPrice'],
                                      self.trade_df['exitPrice'] - self.trade_df['entryPrice'])

    def subset_date_agg_pnl(self):
        self.end_date = self.end_date - timedelta(weeks=self.ph.setup_params.years_to_train*52)
        self.pnl_df = self.pnl_df[self.pnl_df['DateTime'] < self.end_date]
        self.pnl_df = self.pnl_df.groupby(['side', 'paramset_id'], as_index=False).agg(
            row_count=('PnL', 'count'),
            total_pnl=('PnL', 'sum'),
            loss_count=('PnL', lambda x: (x < 0).sum()),
            win_count=('PnL', lambda x: (x > 0).sum()),
            avg_pnl=('PnL', 'mean'),
            avg_pnl_neg=('PnL', lambda x: x[x < 0].mean()),
            avg_pnl_pos=('PnL', lambda x: x[x > 0].mean())
        )
        self.pnl_df.rename(columns={'row_count': 'tot_trades',
                                    'total_pnl': 'PnL'}, inplace=True)

        self.pnl_df['win_percent'] = self.pnl_df['win_count']/self.pnl_df['tot_trades']
        self.pnl_df['expected_value'] = ((self.pnl_df['win_percent'] * self.pnl_df['avg_pnl_pos']) +
                                         ((1 - self.pnl_df['win_percent']) * self.pnl_df['avg_pnl_neg']))
        self.pnl_df['max_potential'] = (((self.pnl_df['win_percent'] * self.pnl_df['avg_pnl_pos']) +
                                        (-(1 - self.pnl_df['win_percent']) * self.pnl_df['avg_pnl_neg'])) *
                                        self.pnl_df['tot_trades'])

    def merge_pnl_params(self):
        self.pnl_df = pd.merge(self.pnl_df, self.param_df, on='paramset_id')
        self.pnl_df.drop_duplicates(subset=['PnL', 'win_percent', 'max_potential'])

    def get_best_param_choice_params(self, even_money):
        if even_money:
            self.pnl_df = self.pnl_df[self.pnl_df['atrLower'] == self.pnl_df['atrUpper']]

        best_params = []
        for side in ['Bull', 'Bear']:
            temp_side = self.pnl_df[self.pnl_df['side'] == side].copy(deep=True)
            temp_side.drop_duplicates(subset=['side', 'PnL'], inplace=True)

            for metric in ['expected_value', 'PnL']:
                pnl_metric = temp_side.sort_values(by=[metric], ascending=False)
                best_params.append(pnl_metric.iloc[0])
                best_params.append(pnl_metric.iloc[-1])

            for metric in ['avg_pnl_neg', 'avg_pnl_pos', 'avg_pnl', 'max_potential', 'win_percent']:
                if metric == 'avg_pnl_neg':
                    pnl_metric = temp_side.sort_values(by=[metric], ascending=True)
                else:
                    pnl_metric = temp_side.sort_values(by=[metric], ascending=False)
                best_params.append(pnl_metric.iloc[0])

        best_params = pd.concat(best_params, axis=1).T
        best_params = best_params.sort_values(by=['PnL'], ascending=False)
        best_params.drop_duplicates(inplace=True)

        self.best_params_df = best_params
        self.valid_params = np.array(best_params['paramset_id'])

    def add_chosen_params(self):
        add_dfs = []
        for side in ['Bull', 'Bear']:
            chosen_params = self.ph.setup_params.chosen_params[side]
            chosen_df = self.pnl_df[self.pnl_df['paramset_id'].isin(chosen_params)]
            chosen_df = chosen_df[chosen_df['side'] == side]
            add_dfs.append(chosen_df)
        add_dfs = pd.concat(add_dfs).reset_index(drop=True)
        self.best_params_df = pd.concat([self.best_params_df, add_dfs]).reset_index(drop=True)
        self.best_params_df.drop_duplicates(inplace=True)

    def save_all_params(self):
        all_params_loc = self.ph.pathmgr.algo_param_data_path('all_params')
        self.pnl_df.to_excel(all_params_loc, index=False)

        best_param_loc = self.ph.pathmgr.algo_param_data_path('best_params')
        self.best_params_df.to_excel(best_param_loc, index=False)

    def adj_lstm_training_nodes(self, ind, model_increase=0):
        size_adj = model_increase * ind / len(self.ph.test_dates) + 1
        for layer in ['lstm_i1_nodes', 'lstm_i2_nodes', 'dense_m1_nodes', 'dense_wl1_nodes']:
            self.ph.ml_model.lstm_dict[layer] = int(self.ph.ml_model.lstm_dict[layer] * size_adj)

    def run_param_chooser(self, even_money=False):
        self.load_files()
        self.set_pnl()
        self.subset_date_agg_pnl()
        self.merge_pnl_params()
        self.get_best_param_choice_params(even_money)
        self.add_chosen_params()
        self.save_all_params()

    def get_valid_params(self, only_chosen=False):
        """TODO: check whether its useful later to keep valid params as a df"""
        valid_params = self.best_params_df[self.best_params_df['side'] == self.ph.side]
        if only_chosen:
            valid_params = (
                valid_params)[(valid_params['side'] == self.ph.side) &
                              (valid_params['paramset_id'].isin(self.ph.setup_params.chosen_params[self.ph.side]))]
        valid_params = valid_params['paramset_id'].to_numpy()
        valid_params = np.sort(valid_params)

        print(f'Training {len(valid_params)} Valid Params: \n'
              f'...{valid_params}')

        return valid_params






