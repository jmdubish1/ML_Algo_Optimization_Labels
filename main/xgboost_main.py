import numpy as np
import pandas as pd
import tensorflow as tf
from ml_tools.process_handler import XgbModelProcessHandler
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_trade_tools import TradeData
from ml_tools.xgboost_model_tools import XgBoostModel
from data_tools.data_prediction_tools import ModelOutputData
from ml_tools.save_handler import SaveHandler
from data_tools.data_mkt_lstm_tools import MLTrainData
from multiprocessing import Process
from data_tools.path_tools import PathManager


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

pd.options.mode.chained_assignment = None  # default='warn'


folder_dict = {
    'model_type': 'xgboost',
    'strategy': 'Double_Candle',
    'security': 'NQ',
    'time_frame_test': '15min',
    'time_frame_train': '5min',
    'time_length': '20years',
    'data_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',

    'trade_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'num_algo_params': '126'
}

setup_dict = {
    'security': 'NQ',
    'other_securities': ['RTY', 'YM', 'ES'],  #, 'GC', 'CL'],
    'sides': ['Bull'],
    'start_train_date': '2010-04-01',
    'final_test_date': '2024-04-01',
    'start_hour': 2,
    'start_minute': 30,
    'test_period_days': 7*13,
    'years_to_train': 3,
    'sample_size': 10,
    'chosen_params': {'Bull': [109, 11, 76, 56, 106, 81, 96, 116, 100, 28],
                      'Bear': [101, 51, 109, 11, 26, 45, 96, 126, 114]},
    'label_classes': ['upper_exit', 'lower_exit', 'time_exit'],

    'over_sample_y': True,
    'train_initial': True,
    'use_best_params': True,
    'retrain_tf': True,
    'randomize_train': True
}


init_xgb_dict = {
    'batch_size': 128,
    'intra_lookback': 16,
    'daily_lookback': 12,
    'n_estimators': np.arange(80, 261, 10),
    'max_depth': np.arange(13, 33, 2),
    'learning_rate': np.arange(.001, .021, .0005),
    'subsample': np.arange(.65, .91, .025),
    'colsample_bytree': np.arange(.65, .91, .025),
    'reg_alpha': [.05, .5, .75, 1., 1.5, 3],
    'reg_lambda': [.05, .25, .5, .75, 1., 1.5, 3],
    'buffer': 500
}


def main():
    ph = XgbModelProcessHandler(setup_dict)
    pm = PathManager(**folder_dict)
    pm.set_pathmgr_ph(ph)
    MktDataSetup(ph)
    SaveHandler(ph)
    AlgoParamResults(ph)
    ph.param_chooser.run_param_chooser()
    
    for ph.side in ph.setup_params.sides:
        TradeData(ph)
        valid_params = ph.param_chooser.get_valid_params()
        
        for ph.paramset_id in valid_params:
            MktDataWorking(ph)
            XgBoostModel(ph, init_xgb_dict)

            ph.xgboost_workflow()


if __name__ == '__main__':
    main()