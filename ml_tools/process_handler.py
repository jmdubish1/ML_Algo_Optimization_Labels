from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
import keras.backend
import pandas as pd
from datetime import timedelta, datetime
import data_tools.general_tools as gt
import cProfile
import pstats
import tensorflow.keras.backend as K
import tensorflow as tf
from numba import cuda
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, AnyStr

from ml_tools.xgboost_model_tools import XgBoostModel, XgbModelData
from ml_tools.lstm_model_tools import LstmModel
from data_tools.data_trade_tools import TradeData
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from ml_tools.save_handler import SaveHandler
from data_tools.data_prediction_tools import ModelOutputData
from data_tools.data_mkt_lstm_tools import MLTrainData
# from ml_tools.autoencoder_model_tools import AutoEncoderModel
from data_tools.path_tools import PathManager


@dataclass
class SetupParams:
    security: str
    other_securities: list
    sides: list
    start_train_date: datetime
    final_test_date: datetime
    start_hour: int
    start_minute: int
    test_period_days: int
    years_to_train: int
    sample_size: int
    chosen_params: dict
    over_sample_y: bool
    retrain_tf: bool
    randomize_train: bool

    label_classes: Optional[list]
    train_initial: Optional[bool]
    use_best_params: Optional[bool]
    predict_tf: Optional[bool]
    use_prev_period_model: Optional[bool]

    num_y_cols: field(init=False) = int

    def __post_init__(self):
        if isinstance(self.start_train_date, str):
            self.start_train_date = pd.to_datetime(self.start_train_date, format='%Y-%m-%d')
        if isinstance(self.final_test_date, str):
            self.final_test_date = pd.to_datetime(self.final_test_date, format='%Y-%m-%d')

        self.num_y_cols = len(self.label_classes)


class ProcessHandler:
    def __init__(self,
                 setup_params: dict):
        self.setup_params = SetupParams(**setup_params)
        self.pathmgr = PathManager
        self.save_handler = SaveHandler
        self.mkt_setup = MktDataSetup
        self.mktdata_working = MktDataWorking
        self.ml_data = MLTrainData
        self.trade_data = TradeData
        self.param_chooser = AlgoParamResults
        self.model_output_data = ModelOutputData

        self.test_dates = self.get_test_dates()
        self.curr_test_date = None

        self.side = str
        self.paramset_id = int

        self.load_current_model = bool
        self.load_previous_model = bool
        self.previous_model_train_path = str
        self.train_modeltf = bool
        self.prior_traintf = bool

    def get_test_dates(self):
        """Gets a list of all test_date's to train. This should go in another class (possibly processHandler)"""
        end_date = pd.to_datetime(self.setup_params.final_test_date, format='%Y-%m-%d')
        end_date = gt.ensure_friday(end_date)
        start_date = end_date - timedelta(weeks=self.setup_params.years_to_train*52)

        test_dates = []
        current_date = start_date
        while current_date <= end_date:
            test_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=self.setup_params.test_period_days)

        return test_dates

    def reset_gpu_memory(self):
        tf.keras.backend.clear_session()
        gc.collect()


class LstmProcessProcessHandler(ProcessHandler):
    def __init__(self, setup_params: dict):
        super().__init__(setup_params)
        self.ml_model = LstmModel

    def decide_model_to_train(self, test_date, use_previous_model):
        """Used only for Neural Networks"""
        current_model_exists = os.path.exists(f'{self.save_handler.model_save_path}\\model.keras')
        previous_model_exists = os.path.exists(f'{self.save_handler.previous_model_path}\\model.keras')
        self.prior_traintf = False
        self.load_current_model = False
        self.load_previous_model = False

        if current_model_exists:
            print(f'Retraining Model: {self.save_handler.model_save_path}')
            self.prior_traintf = True
            self.load_current_model = True
            self.previous_train_path = self.save_handler.model_save_path

            if not self.setup_params.retrain_tf:
                print(f'Predicting only: {self.save_handler.model_save_path}')
                self.train_modeltf = False

        elif previous_model_exists and use_previous_model:
            print(f'Training model from previous model: {self.save_handler.previous_model_path}')
            self.prior_traintf = True
            self.train_modeltf = True
            self.load_previous_model = True
            self.previous_train_path = self.save_handler.previous_model_path

        else:
            print(f'Training New Model...')
            self.train_modeltf = True
            self.prior_traintf = False
        print(f'Training Model: \n...Param: {self.paramset_id} \n...Side: {self.side} \n...Test Date: {test_date}')

    def decide_load_prior_model(self):
        if self.prior_traintf:
            print(f'Loading Prior Model: {self.previous_train_path}')
            if self.load_current_model:
                self.save_handler.load_lstm_model(self.save_handler.model_save_path)
            elif self.load_previous_model:
                self.save_handler.load_lstm_model(self.save_handler.previous_model_path)

    def decide_load_scalers(self):
        load_scalers = False
        if self.prior_traintf:
            load_scalers = True
            self.save_handler.load_scalers(self.setup_params.retrain_tf)

        else:
            print('Creating New Scalers')

        return load_scalers

    def work_single_lstm_model(self, test_date, use_prev_period_model, predict_tf, ind, train_dict):
        print(f'Modelling {test_date}')
        self.trade_data.set_dates(test_date)
        self.trade_data.create_working_df()
        self.save_handler.set_model_train_paths_lstm()
        self.trade_data.separate_train_test()

        self.decide_model_to_train(test_date, use_prev_period_model)
        keras.backend.clear_session()
        if self.train_modeltf or predict_tf:
            load_scalers = self.decide_load_scalers()
            self.ml_data.prep_train_test_data(load_scalers)
            self.decide_load_prior_model()
            self.param_chooser.adj_lstm_training_nodes(ind, model_increase=.05)
            if self.train_modeltf:
                self._train_lstm_model(ind, train_dict['randomize_train'])

            self.save_handler.load_lstm_model(self.save_handler.model_save_path)
            self.model_output_data.predict_lstm_data(ind, use_opt_thres=True)
            predicted_data = self.model_output_data.prediction_analysis()
            self.save_handler.new_save_metrics(test_date, predicted_data)
            self.reset_gpu_memory()

    def _train_lstm_model(self, ind, randomize_tf):
        if not self.prior_traintf:
            self.ml_model.build_compile_model()
        else:
            print(f'Loaded Previous Model')
        self.ml_model.train_model(randomize_tf)

        self.save_handler.save_lstm_model(ind)
        self.save_handler.save_scalers()


class XgbModelProcessHandler(ProcessHandler):
    def __init__(self, setup_params: dict):
        super().__init__(setup_params)
        self.ml_model = XgBoostModel
        self.uniq_id = str
        self.train_modeltf = bool
        self.previous_train_path = None

    def xgboost_workflow(self):
        if self.setup_params.train_initial:
            self.test_dates = self.test_dates[0]
        for test_date in self.test_dates:
            self.pre_training_work_xgb(test_date)
            if self.setup_params.train_initial:
                if self.setup_params.use_best_params:
                    self.ml_model.get_best_params()
                    self.ml_model.create_best_retrain_dict()
                    self.ml_model.get_param_combos()
            else:
                self.ml_model.get_best_params()
                self.ml_model.create_eval_param_dict()

            self.ml_model.xgboost_model_training(test_date)

            self.reset_gpu_memory()

    def pre_training_work_xgb(self, test_date):
        self.trade_data.set_dates(test_date)
        self.trade_data.create_working_df()
        self.save_handler.set_model_train_paths_xgb()
        self.trade_data.separate_train_test()
        self.ml_data.prep_train_test_data()
        self.set_xgb_param_dict()

    def set_xgb_param_dict(self):
        if self.ml_model.use_best_params:
            xgb_params = self.save_handler.load_xgboost_params()
            self.ml_model.find_best_params(xgb_params)
        self.ml_model.create_gridsearch_dict()



