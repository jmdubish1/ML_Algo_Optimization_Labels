import os
import numpy as np
import pandas as pd
import xgboost as xgb
import itertools
import time
import random
import cupy as cp
from sklearn.metrics import roc_auc_score, precision_score
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple
import ml_tools.general_ml_tools as glt
import datetime
from ml_tools.general_ml_tools import load_full_cupy_dataset, BufferedBatchGenerator
from ml_tools.loss_functions import xgb_weighted_auc, xgb_weighted_precision
import hashlib

if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler



@dataclass
class XgbModelData:
    batch_s: int
    buffer: int
    daily_len: int
    intra_len: int
    use_best_params: bool

    n_estimators: np.array
    max_depth: np.array
    learning_rate: np.array
    subsample: np.array
    colsample_bytree: np.array
    reg_alpha: List
    reg_lambda: List
    buffer: int
    param_grid: dict = field(init=False)
    tree_params: dict = field(init=False)

    param_combos: Optional[List]
    param_dicts: Optional[List[Dict]]
    class_weights: Optional[List[float]] = None
    uniq_id: Optional[str] = None

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

    def __init__(self, param_dict: Dict):
        for key, val in param_dict.items():
            setattr(self, key, val)


class XgBoostModel:
    def __init__(self,
                 process_handler: ProcessHandler,
                 init_xgb_dict: dict
                 ):
        self.ph = process_handler
        self.ph.ml_model = self
        self.model_data = XgbModelData(**init_xgb_dict)
        self.model: xgb.Booster = None
        self.param_combos = []
        self.param_dicts = []
        self.tested_params = []
        self.testing_params = None

    def load_dmats(self) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """Load training and validation datasets."""
        train_gen = BufferedBatchGenerator(self.ph, self.model_data.buffer, train=True,
                                           randomize=self.ph.setup_params.randomize_data, dat_type='numpy')
        test_gen = BufferedBatchGenerator(self.ph, self.model_data.buffer, train=False, dat_type='numpy')

        x_train, y_train = load_full_cupy_dataset(train_gen)
        x_val, y_val = load_full_cupy_dataset(test_gen)

        y_train_weights = set_y_class_weights(y_train, self.ph.ml_data.class_weights)
        y_val_weights = set_y_class_weights(y_val, self.ph.ml_data.class_weights)

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=y_train_weights)
        dval = xgb.DMatrix(x_val, label=y_val, weight=y_val_weights)
        return dtrain, dval

    def evaluate_param_combo(self, params: Dict,
                             dtrain: xgb.DMatrix,
                             dval: xgb.DMatrix,
                             test_date: datetime,
                             ind: int):
        """Evaluate a single parameter combination."""
        start = time.time()
        param_dict = dict(zip(self.model_data.param_grid.keys(), params))
        uniq_id = generate_unique_id(param_dict, self.ph.paramset_id)
        n_estimators = int(param_dict.pop('n_estimators'))

        if not self.ph.setup_params.train_initial and os.path.exists(self.ph.pathmgr.model_path(uniq_id, test_date)):
            self.ph.save_handler.load_xgb_model(uniq_id)
        else:
            self.model = xgb.train(
                params=param_dict,
                dtrain=dtrain,
                evals=[(dval, "Validation")],
                num_boost_round=n_estimators,
                verbose_eval=False,
                custom_metric=xgb_weighted_auc,
                maximize=True,
            )

        self.ph.save_handler.save_xgb_model(uniq_id)

        y_pred_proba = self.model.predict(dval)
        auc_score_train = xgb_weighted_auc(y_pred_proba, dtrain)
        prec_score_train = xgb_weighted_precision(y_pred_proba, dtrain)

        auc_score_val = xgb_weighted_auc(y_pred_proba, dval)
        prec_score_val = xgb_weighted_precision(y_pred_proba, dval)

        param_dict.update({
            "n_estimators": n_estimators,
            "AUC_train": round(auc_score_train, 5),
            "Precision_train": round(prec_score_train, 5),
            "AUC_test": round(auc_score_val, 5),
            "Precision_test": round(prec_score_val, 5),
            "Model_param_id": uniq_id,
            "Time": round(time.time() - start, 2),
        })
        print(f"Params: {param_dict}\n{ind}/{len(self.param_combos)}")
        self.tested_params.append(param_dict)

    def xgboost_model_training(self, test_date):
        dtrain, dval = self.load_dmats()
        [self.evaluate_param_combo(params, dtrain, dval, test_date, ind)
         for ind, params in enumerate(self.param_dicts)]

        self.ph.save_handler.save_xgboost_params(self.param_dicts)
        best_params = max(self.param_dicts, key=lambda x: x['AUC'])
        print_params = \
            {key: value for key, value in best_params.items() if key not in ['probability_preds', 'preds_1hot']}
        print(f"Best Params: {print_params}, Best AUC: {best_params['AUC']}")

        return best_params, self.param_dicts

    def get_best_params(self, n_best=5):
        xgb_params = self.ph.save_handler.load_xgboost_params()
        testing_params = xgb_params[(xgb_params['cluster'] == 'tgt') &
                                    (xgb_params['paramset_id'] == self.ph.paramset_id)]
        testing_params = testing_params[pd.notna(testing_params['unique_id'])]
        testing_params = testing_params.sort_values(by='Precision', ascending=False)
        self.testing_params = testing_params.iloc[:n_best]

    def get_param_combos(self):
        all_combos = list(itertools.product(*self.model_data.param_grid.values()))
        self.param_dicts = random.sample(all_combos, self.ph.setup_params.sample_size)

    def create_best_retrain_dict(self):
        for key, val in self.model_data.param_grid.items():
            if (not isinstance(val, str)) and (key in self.testing_params.columns) and (key not in ['num_class']):
                min_val, max_val, std_val = (self.testing_params[key].min(), self.testing_params[key].max(),
                                             self.tested_params[key].std() * 1.5)
                self.model_data.param_grid[key] = np.linspace(min_val - std_val, max_val + std_val, 5).tolist()

    def create_eval_param_dict(self):
        self.param_dicts = []
        for ind, row in self.testing_params:
            temp_dict = {}
            for key, val in self.model_data.param_grid.items():
                if (not isinstance(val, str)) and (key in self.testing_params.columns) and (key not in ['num_class']):
                    temp_dict[key] = row[key]
            self.param_dicts.append(temp_dict)


def get_random_param_combos(param_grid, n_samples):
    all_combos = list(itertools.product(*param_grid.values()))
    return random.sample(all_combos, n_samples)


def set_y_class_weights(cupy_arr: cp.ndarray, class_weights: List[float]) -> cp.ndarray:
    """Set class weights for samples."""
    class_weights_cp = cp.array(class_weights)
    return class_weights_cp[cp.asarray(cupy_arr, dtype=cp.int32)]


def generate_unique_id(data_dict, paramset_id):
    """
    Generate a unique ID based on a dictionary containing `paramset_id` and XGBoost params.

    Parameters:
        data_dict (dict): A dictionary with keys such as `paramset_id`, `max_depth`, etc.
        existing_ids (set): A set of IDs to ensure uniqueness.
    Returns:
        str: A unique ID.
    """
    unique_string = (
        f"{data_dict['max_depth']}_"
        f"{data_dict['learning_rate']}_"
        f"{data_dict['subsample']}_"
        f"{data_dict['colsample_bytree']}_"
        f"{data_dict['reg_alpha']}_"
        f"{data_dict['reg_lambda']}_"
        f"{data_dict['n_estimators']}"
    )

    hashed_id = hashlib.md5(unique_string.encode()).hexdigest()[:12]
    full_id = f"Algo_{paramset_id}_{hashed_id}"

    return full_id