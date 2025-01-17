import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple, AnyStr
from datetime import datetime

if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler

@dataclass
class PathManager:
    model_type: str
    strategy: str
    security: str
    mkt_dat_loc: str
    algo_dat_loc: str
    time_frame_test: str
    time_frame_train: str
    time_len: str
    num_algo_params: str

    ph: Optional[None] = None
    # test_date: Optional[datetime] = None
    # ml_dat_loc: str = field(init=False)
    # ml_side_loc: str = field(init=False)
    agg_ffd_loc: Optional[str] = None
    algo_side_loc: Optional[str] = None

    def __post_init__(self):
        self.agg_ffd_loc = os.path.join(self.algo_dat_loc, 'agg_data', 'all_FFD_params.xlsx')

        self.algo_dat_loc = (
            os.path.join(self.algo_dat_loc,
                         self.security,
                         self.time_frame_test,
                         f'{self.time_frame_test}_test_{self.time_len}'))

    def set_algo_side_loc(self):
        self.algo_side_loc = os.path.join(self.algo_dat_loc, self.model_type, self.ph.side)
        self.ph.save_handler.check_create_model_folder()

    def set_pathmgr_ph(self, ph):
        self.ph = ph
        ph.pathmgr = self

    def mtk_data_path(self, security: str):
        return os.path.join(self.mkt_dat_loc, f'{security}_{self.time_frame_train}_20240505_20040401.txt')

    def trade_data_path(self):
        return os.path.join(self.algo_dat_loc,
                            f'{self.security}_{self.time_frame_test}_{self.strategy}_'
                            f'{self.num_algo_params}_trades.feather')

    def trade_params_path(self):
        return os.path.join(self.algo_dat_loc,
                            f'{self.security}_{self.time_frame_test}_{self.strategy}_'
                            f'{self.num_algo_params}_params.feather')

    def model_path(self, uniq_id: int,
                   test_date: datetime,
                   extension="json"):
        if not isinstance(test_date, str):
            test_date = test_date.strftime('%Y-%m-%d')

        return os.path.join(self.algo_side_loc, 'Models', str(self.ph.paramset_id),
                            f'{uniq_id}_{test_date}.{extension}')

    def model_data_path(self, file_name: str):
        data_path = os.path.join(self.algo_side_loc, 'Data', 'validation', str(self.ph.paramset_id),
                                 f'{self.ph.test_date}_{file_name}.xlsx')
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        return data_path

    def algo_param_data_path(self, file_name: str):
        return os.path.join(self.algo_dat_loc, f'{self.security}_{self.time_frame_test}_{file_name}.xlsx')

    def xgb_param_data_path(self, all_params=False):
        paramset_id = 'all' if all_params else str(self.ph.paramset_id)
        if all_params:
            data_file = os.path.join(self.algo_side_loc, 'Data', str(paramset_id),
                                     'pred_xgb_all_params.xlsx')
        else:
            data_file = os.path.join(self.algo_side_loc, 'Data', str(paramset_id),
                                     f'{paramset_id}_xgb_best_params.xlsx')
        return data_file



