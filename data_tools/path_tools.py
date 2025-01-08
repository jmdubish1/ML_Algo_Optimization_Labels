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
    algo_dat_loc: str
    mkt_dat_loc: str
    time_frame_test: str
    time_frame_train: str
    time_len: str
    num_algo_params: str

    paramset_id: Optional[int]
    test_date: Optional[datetime]
    side: Optional[str]
    algo_dat_loc: str = field(init=False)
    algo_side_loc: str = field(init=False)
    ph: Optional[ProcessHandler] = ProcessHandler

    def __post_init__(self):
        self.algo_dat_loc = (
            os.path.join(self.algo_dat_loc,
                         self.security,
                         self.time_frame_test,
                         f'{self.time_frame_test}_test_{self.time_len}'))

        self.algo_side_loc = (
            os.path.join(self.algo_dat_loc,
                         self.security,
                         self.time_frame_test,
                         f'{self.time_frame_test}_test_{self.time_len}',
                         self.model_type,
                         self.side))

    def set_pathmgr_ph(self, ph):
        self.test_date = ph.test_date
        self.side = ph.side
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

    def model_path(self, uniq_id: int, test_date: datetime, extension="bin"):
        test_date = test_date.strftime('%Y-%m-%d')
        return os.path.join(self.algo_dat_loc, self.model_type, 'Models', self.ph.paramset_id,
                            f'{uniq_id}_{test_date}.{extension}')

    def model_data_path(self, uniq_id: int, file_name: str):
        test_date = self.test_date.strftime('%Y-%m-%d')
        return os.path.join(self.algo_dat_loc, self.model_type, 'Data', self.ph.paramset_id,
                            f'{uniq_id}_{test_date}_{file_name}.xlsx')

    def algo_param_data_path(self, file_name: str):
        return os.path.join(self.algo_dat_loc, f'{self.security}_{self.time_frame_test}_{file_name}.xlsx')

    def xgb_param_data_path(self, all_params=False):
        paramset_id = 'all' if all_params else self.ph.paramset_id
        return os.path.join(self.algo_side_loc, 'Data', paramset_id, f'{paramset_id}_xgb_best_params.xlsx')



