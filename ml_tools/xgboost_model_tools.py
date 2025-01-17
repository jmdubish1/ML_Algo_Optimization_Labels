import os
import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
import time
import random
import cupy as cp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
import ml_tools.general_ml_tools as glt
import datetime
from ml_tools.general_ml_tools import load_full_cupy_dataset, BufferedBatchGenerator
from ml_tools.loss_functions import xgb_weighted_auc, xgb_weighted_precision, xgb_weighted_f1beta
from data_tools import general_tools as gt

if TYPE_CHECKING:
    from ml_tools.process_handler import XgbModelProcessHandler


@dataclass
class XgbModelData:
    batch_size: int
    buffer: int
    daily_len: int
    intra_len: int

    # Tree parameters
    n_estimators: np.array
    max_depth: np.array
    learning_rate: np.array
    subsample: np.array
    colsample_bytree: np.array
    reg_alpha: List[float]
    reg_lambda: List[float]

    # Initialized dynamically
    param_grid: dict = field(init=False)
    tree_params: dict = field(init=False)

    # Optional parameters
    param_combos: Optional[List] = field(default_factory=list)
    param_dicts: Optional[List[Dict]] = field(default_factory=list)
    class_weights: Optional[List[float]] = None
    uniq_id: Optional[str] = None
    use_tgt_param: Optional[bool] = False

    def __post_init__(self):
        self.param_grid = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'objective': ['multi:softprob'],
            'num_class': [3],
            'tree_method': ['hist'],
            'device': ['cuda'],
        }


class XgBoostModel:
    def __init__(self,
                 process_handler: "XgbModelProcessHandler",
                 init_xgb_dict: dict
                 ):
        self.ph = process_handler
        self.ph.ml_model = self
        self.model_data = XgbModelData(**init_xgb_dict)
        self.model: xgb.Booster = None
        self.param_combos = []
        self.param_dicts = []
        self.tested_params = []
        self.testing_params = pd.DataFrame
        self.input_shape = None
        self.class_weights = None

    def load_dmats(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """Load training and validation datasets."""
        self.ph.save_handler.set_model_train_paths_xgb()
        train_gen = BufferedBatchGenerator(self.ph, self.model_data.buffer, train=True,
                                           randomize=self.ph.setup_params.randomize_train, dat_type='numpy')
        test_gen = BufferedBatchGenerator(self.ph, self.model_data.buffer, train=False, dat_type='numpy')

        x_train, y_train = load_full_cupy_dataset(train_gen)
        x_val, y_val = load_full_cupy_dataset(test_gen)

        _, self.class_weights = glt.get_class_weights(self.ph)
        y_train_weights = set_y_class_weights(y_train, self.class_weights)
        y_val_weights = set_y_class_weights(y_val, self.class_weights)

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=y_train_weights)
        dval = xgb.DMatrix(x_val, label=y_val, weight=y_val_weights)
        print(dtrain.get_weight())
        return dtrain, dval

    def evaluate_param_combo(self, params: Dict,
                             dtrain: xgb.DMatrix,
                             dval: xgb.DMatrix,
                             test_date: datetime,
                             ind: int):
        """Evaluate a single parameter combination."""

        start = time.time()
        if not isinstance(params, dict):
            param_dict = dict(zip(self.model_data.param_grid.keys(), params))
        else:
            param_dict = params
        uniq_id = gt.generate_unique_id(param_dict, self.ph.paramset_id)
        n_estimators = param_dict.pop('n_estimators')

        print(uniq_id)
        evals_result = {}

        if os.path.exists(self.ph.pathmgr.model_path(uniq_id, test_date)):
            self.ph.save_handler.load_xgb_model(uniq_id)

        else:
            self.model = xgb.train(
                params=param_dict,
                dtrain=dtrain,
                evals=[(dtrain, "Train"), (dval, "Validation")],
                num_boost_round=n_estimators,
                verbose_eval=False,
                evals_result=evals_result,
                custom_metric=xgb_weighted_f1beta,
                maximize=True,
            )

        self.ph.save_handler.save_xgb_model(uniq_id)

        f1_score_train = evals_result["Train"]["weighted_f1beta"][-1]
        f1_score_val = evals_result["Validation"]["weighted_f1beta"][-1]

        prec_score_train = xgb_weighted_precision(self.model.predict(dtrain), dtrain)[1]

        y_pred_proba = self.model.predict(dval)
        prec_score_val = xgb_weighted_precision(y_pred_proba, dval)[1]

        param_dict.update({
            "n_estimators": n_estimators,
            "F1_train": round(f1_score_train, 5),
            "Precision_train": round(prec_score_train, 5),
            "F1_test": round(f1_score_val, 5),
            "Precision_test": round(prec_score_val, 5),
            "Model_param_id": uniq_id,
            "Time": round(time.time() - start, 2),
            "Pred_probs": y_pred_proba,
        })

        if self.ph.setup_params.use_best_params:
            param_dict['tgt_param'] = 1

        data_print = ", ".join(f"{key}: {value}" for key, value in param_dict.items() if key != "Pred_probs")
        print(f"Params: {self.ph.paramset_id}: {data_print}\n{ind}/{self.ph.setup_params.sample_size}")
        self.tested_params.append(param_dict)

        if ind % 25 == 0:
            self.ph.save_handler.save_xgboost_params(self.tested_params)

    def xgboost_model_training(self, test_date):

        dtrain, dval = self.load_dmats()
        [self.evaluate_param_combo(params, dtrain, dval, test_date, ind)
         for ind, params in enumerate(self.param_dicts)]

        self.ph.save_handler.save_xgboost_probs()
        self.ph.save_handler.save_xgboost_params(self.tested_params)
        best_params = max(self.tested_params, key=lambda x: x['F1_test'])
        print_params = \
            {key: value for key, value in best_params.items() if key not in ['probability_preds', 'preds_1hot']}
        print(f"Best Params: {print_params}, Best Weighted F1: {best_params['F1_test']}")

        return best_params, self.param_dicts

    def get_best_params(self, full_test=False, n_best=20):
        xgb_params = self.ph.save_handler.load_xgb_params(all_params=True)
        testing_params = xgb_params[(xgb_params['paramset_id'] == self.ph.paramset_id)]
        # testing_params = testing_params[testing_params['Precision_test'].isna() |
        #                                 testing_params['Precision_test'] == '']
        if full_test:
            n_best = 5
        testing_params = testing_params.sort_values(by='Precision_pred', ascending=False)
        self.testing_params = testing_params.iloc[:n_best].reset_index(drop=True)

    def get_param_combos(self):
        all_combos = list(itertools.product(*self.model_data.param_grid.values()))
        self.param_dicts = random.sample(all_combos, self.ph.setup_params.sample_size)

    def create_best_retrain_dict(self):
        for key, val in self.model_data.param_grid.items():
            if ((not isinstance(self.testing_params.loc[0, key], str)) and
                    (key in self.testing_params.columns) and
                    (key not in ['num_class'])):
                test_arr = self.testing_params.loc[:, key].to_numpy()
                mean_val = np.mean(test_arr)
                std_val = np.std(test_arr)
                min_val = max(np.min(test_arr), mean_val - std_val)
                max_val = min(np.max(test_arr), mean_val + std_val)
                new_range = np.linspace(min_val, max_val, 6)

                if key in ['n_estimators', 'max_depth']:
                    new_range[new_range < 4] = 4
                    new_range = new_range.astype(int).tolist()
                else:
                    new_range[new_range <= 0] = 0.0001
                    new_range = [round(i, 5) for i in new_range]
                self.model_data.param_grid[key] = new_range

    def create_eval_param_dict(self):
        all_params = [key for key in self.model_data.param_grid.keys()]
        self.param_dicts = self.testing_params[all_params].to_dict(orient='records')


def set_y_class_weights(cupy_arr: cp.ndarray, class_weights: List[float]) -> cp.ndarray:
    """Set class weights for samples."""
    class_weights_cp = cp.array(class_weights)
    return class_weights_cp[cp.asarray(cupy_arr, dtype=cp.int32)]



