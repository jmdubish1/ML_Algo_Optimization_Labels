import numpy as np
import pandas as pd
import tensorflow as tf
from ml_tools.process_handler import XgbModelProcessHandler
from data_tools.data_mkt_setup_tools import MktDataSetup, MktDataWorking
from analysis_tools.param_chooser import AlgoParamResults
from data_tools.data_trade_tools import TradeData
from ml_tools.xgboost_model_tools import XgBoostModel
# from data_tools.data_prediction_tools import ModelOutputData
from ml_tools.save_handler import SaveHandler
from data_tools.data_mkt_ml_tools import MLTrainData
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
    'time_len': '20years',
    'mkt_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data',
    'algo_dat_loc': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR',
    'num_algo_params': '126'
}

setup_dict = {
    'security': 'NQ',
    'other_securities': ['RTY', 'YM', 'ES'],  #, 'GC', 'CL'],
    'sides': ['Bull', 'Bear'],
    'start_train_date': '2010-04-01',
    'final_test_date': '2024-04-01',
    'start_hour': 2,
    'start_minute': 30,
    'test_period_days': 7*13,
    'years_to_train': 3,
    'sample_size': 50,

    'chosen_params': {'Bull': [109, 126, 76, 51, 106, 81, 96, 116, 101, 21],
                      'Bear': [101, 51, 109, 11, 26, 45, 96, 126, 114, 46]
                      },
    'label_classes': ['upper_exit', 'lower_exit', 'time_exit'],

    'over_sample_y': True,
    'train_initial': False,
    'use_best_params': True,
    'retrain_tf': True,
    'randomize_train': True
}


init_xgb_dict = {
    'batch_size': 128,
    'buffer': 50,
    'daily_len': 12,
    'intra_len': 32,
    'n_estimators': np.concatenate([np.arange(4, 26, 3), np.arange(30, 241, 15)]),
    'max_depth': np.arange(3, 32, 2),
    'learning_rate': np.arange(.001, .021, .0005),
    'subsample': np.arange(.65, .91, .025),
    'colsample_bytree': np.arange(.65, .91, .025),
    'reg_alpha': [.005, .05, .5, 1., 1.5, 3, 5, 8, 10],
    'reg_lambda': [.005, .05, .5, 1., 1.5, 3, 5, 8, 10],
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
        ph.pathmgr.set_algo_side_loc()
        TradeData(ph)
        valid_params = ph.param_chooser.get_valid_params(only_chosen=True)
        
        for ph.paramset_id in valid_params[::-1]:
            MktDataWorking(ph)
            MLTrainData(ph)
            XgBoostModel(ph, init_xgb_dict)
            ph.xgboost_workflow()


if __name__ == '__main__':
    main()