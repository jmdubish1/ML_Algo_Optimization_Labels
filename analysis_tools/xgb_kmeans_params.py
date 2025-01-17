import os
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import hashlib
from data_tools import general_tools as gt

os.environ["OMP_NUM_THREADS"] = "1"


class ParamsetResults:
    def __init__(self,
                 setup_dict: dict,
                 param_df: pd.DataFrame,
                 paramset_id,
                 file,
                 use_tgt_param: bool):
        self.paramset_id = paramset_id
        self.side = setup_dict['side']
        self.file_folder = os.path.dirname(file)
        self.param_df = param_df
        self.param_test_df = pd.DataFrame
        self.summary_df = pd.DataFrame
        self.pca_df = pd.DataFrame
        self.data_cols = setup_dict['data_cols']
        self.n_clusters = setup_dict['n_clusters']
        self.tgt_ranges = None
        self.tgt_cluster = int
        self.use_tgt_param = use_tgt_param

    def build_clusters(self):
        self.param_df['paramset_id'] = self.paramset_id
        prec_col = None
        if 'Precision_test' in self.param_df.columns:
            prec_col = 'Precision_test'
        else:
            prec_col = 'Precision_pred'
        df = self.param_df[self.param_df[prec_col] != '']
        df_tgt = None

        if self.use_tgt_param:
            df, df_tgt = separate_given_tgt_dfs(df)
        elif self.tgt_ranges:
            df, df_tgt = separate_tgt_dfs(df, self.tgt_ranges)

        df = kmeans_work(df, self.data_cols, self.n_clusters)

        if df_tgt is not None:
            df = pd.concat([df, df_tgt])

        self.param_df = df

    def build_paramset_test_set(self):
        df = self.param_df[self.data_cols]
        param_dict = {
            col: np.round(np.linspace(df[col].min(), df[col].max(), 50), 3) for col in df.columns
        }

        test_param_df = {
            col: np.random.choice(values, 5000) for col, values in param_dict.items()
        }

        test_param_df = pd.DataFrame(test_param_df)
        for col in ['n_estimators', 'max_depth']:
            test_param_df[col] = test_param_df[col].to_numpy().astype(int)

        self.param_test_df = test_param_df

    def xgboost_model(self):
        self.build_paramset_test_set()
        x_train, x_test, y_train, y_test = train_test_split(self.param_df[self.data_cols],
                                                            self.param_df['Precision_test'],
                                                            test_size=0.2)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        xgb_model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        dtest = xgb.DMatrix(self.param_test_df[self.data_cols])
        y_pred = xgb_model.predict(dtest)
        self.param_test_df['Precision_pred'] = y_pred
        self.param_test_df = self.param_test_df.sort_values(by='Precision_pred', ascending=False)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.param_test_df.head(10))
        self.param_test_df = self.param_test_df.head(50)
        self.param_test_df['n_estimators'] = np.random.randint(10, 250, size=len(self.param_test_df))
        self.param_test_df['is_tgt'] = True
        self.param_test_df['paramset_id'] = self.paramset_id
        self.param_test_df['objective'] = 'multi:softprob'
        self.param_test_df['tree_method'] = 'hist'
        self.param_test_df['device'] = 'cuda'
        self.param_test_df['num_class'] = 3

    def create_cluster_summary(self):
        work_cols = ['Precision_test', 'Time'] + self.data_cols
        cluster_summary = self.param_df.groupby('cluster')[work_cols].agg(['mean', 'std', 'min', 'max', 'count'])
        self.summary_df = cluster_summary.sort_values(('Precision_test', 'mean'), ascending=False)
        self.tgt_cluster = self.summary_df.index[0]

    def create_tgt_ranges(self, use_given_tgt=True):
        self.tgt_ranges = {
            'n_estimators': [],
            'max_depth': [],
            'learning_rate': [],
            'subsample': [],
            'colsample_bytree': [],
            'reg_alpha': [],
            'reg_lambda': [],
            'objective': ['multi:softprob'],
            'num_class': [3],
            'eval_metric': ['auc'],
            'tree_method': ['hist'],
            'device': ['cuda']
        }

        if use_given_tgt:
            work_df = self.param_df[self.param_df['tgt_param'] == 1]
            for col in self.data_cols:
                mean = np.mean(work_df[col])
                std = np.std(work_df[col])
                low = mean - std * 1.5
                high = mean + std * 1.5

                new_range = np.round(np.linspace(low, high, 5), 5)

                if col in ['n_estimators', 'max_depth']:
                    new_range = new_range.astype(int)

                new_range[new_range <= 0] = 0.0001
                self.tgt_ranges[col] = new_range
                self.tgt_cluster = 'tgt'
        else:
            self.param_df = self.param_df.sort_values(by='Precision_test', ascending=False)
            work_df = self.param_df[self.param_df['cluster'] == self.summary_df.index[0]]
            for col in self.data_cols:
                temp_df = work_df.groupby('cluster')[col].agg(['mean', 'std'])
                mean = temp_df['mean'].iloc[0]
                std = temp_df['std'].iloc[0]
                low = mean - std * 1.5
                high = mean + std * 1.5

                new_range = np.round(np.linspace(low, high, 5), 5)

                if col in ['n_estimators', 'max_depth']:
                    new_range = new_range.astype(int)

                new_range[new_range <= 0] = 0.0001
                self.tgt_ranges[col] = new_range

    def grow_tgt_cluster(self, use_given_tgt=True):
        if not self.tgt_ranges:
            self.create_tgt_ranges(use_given_tgt)

        tgt_cluster_mask = np.ones_like(self.param_df['cluster'].values)

        for col in self.data_cols:
            col_mask = ((self.tgt_ranges[col][0] <= self.param_df[col]) &
                        (self.param_df[col] <= self.tgt_ranges[col][-1]))
            tgt_cluster_mask = col_mask * tgt_cluster_mask

        self.param_df.loc[tgt_cluster_mask, 'cluster'] = self.tgt_cluster

    def plot_cluster_data(self, final_pred=False):
        if self.use_tgt_param:
            self.param_df['is_tgt'] = self.param_df['tgt_param'] == 1
        else:
            self.param_df['is_tgt'] = self.param_df['cluster'] == self.tgt_cluster
        self.param_df.reset_index(drop=True, inplace=True)
        self.param_df['cluster'] = self.param_df['cluster'].astype(str)

        test_cols = ['F1_test', 'Precision_test']
        if final_pred:
            test_cols = ['Precision_pred']

        for test_col in test_cols:
            for col in self.data_cols:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    data=self.param_df, x=col, y=test_col,
                    hue='cluster', palette='tab10',
                    style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
                )
                sns.regplot(data=self.param_df, x=col, y=test_col, scatter=False, color='darkred', order=3)
                plt.title(f'KMeans {test_col} for {col}')
                plt.savefig(f'{self.file_folder}\\{self.paramset_id}_{col}_kmeans_{test_col}.png')
                plt.close()

                if not final_pred:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(
                        data=self.param_df, x=col, y='Time',
                        hue='cluster', palette='tab10',
                        style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
                    )

                    sns.regplot(data=self.param_df, x=col, y='Time', scatter=False, color='darkred', order=3)
                    plt.title(f'KMeans Time for {col}')
                    plt.savefig(f'{self.file_folder}\\{self.paramset_id}_{col}_kmeans_Time.png')
                    plt.close()

            if not final_pred:
                # F1_test vs Precision
                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    data=self.param_df, x='F1_test', y='Precision_test',
                    hue='cluster', palette='tab10',
                    style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
                )
                sns.regplot(data=self.param_df, x='F1_test', y='Precision_test', scatter=False, color='darkred', order=3)
                plt.title(f'KMeans F1_test vs. Precision_test')
                plt.savefig(f'{self.file_folder}\\{self.paramset_id}_F1_testPrec.png')
                plt.close()

                # Prec vs. Cluster
                plt.figure(figsize=(8, 6))
                sns.scatterplot(
                    data=self.param_df, x='cluster', y=test_col,
                    hue='cluster', palette='tab10',
                    style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
                )
                plt.title(f'KMeans cluster vs. Precision_test')
                plt.savefig(f'{self.file_folder}\\{self.paramset_id}_clusterPrec.png')
                plt.close()

    def save_dfs(self):
        save_file = f'{self.file_folder}\\{self.paramset_id}_summary_stats.xlsx'
        self.summary_df.to_excel(save_file)


def kmeans_work(df_, relevant_cols_, n_clusters):
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df_[relevant_cols_])

    imputer = SimpleImputer(strategy="mean")
    df_normalized = imputer.fit_transform(df_normalized)

    kmeans = KMeans(n_clusters=n_clusters)
    df_['cluster'] = kmeans.fit_predict(df_normalized)

    return df_


def separate_tgt_dfs(df, target_dict):
    tgt_mask = df.apply(
        lambda row: all(
            target_dict[col][0] <= row[col] <= target_dict[col][1]
            for col in target_dict.keys()
        ), axis=1
    )

    df_tgt_ = df[tgt_mask].reset_index(drop=True)
    df_else = df[~tgt_mask].reset_index(drop=True)

    return df_else, df_tgt_


def separate_given_tgt_dfs(df):
    if 'tgt_param' not in df.columns:
        df['tgt_param'] = 0
    tgt_mask = (df['tgt_param'] == 1)

    df_tgt = df[tgt_mask].reset_index(drop=True)
    df_tgt['cluster'] = 'tgt'
    df_else = df[~tgt_mask].reset_index(drop=True)

    return df_else, df_tgt


def get_excel_files(parent_folder):
    excel_files = []

    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            match = re.search(r"(\d+)_xgb_best_params\.xlsx", file)
            if match and int(match.group(1)):
                excel_files.append(os.path.join(root, file))

    return excel_files


def save_all_params(df, file_path, pred_tf=False, combined_test=None):
    match_cols = ['max_depth', 'subsample', 'colsample_bytree', 'learning_rate', 'reg_alpha', 'reg_lambda',
                  'n_estimators']
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=match_cols)
    else:
        combined_df = df

    sort_col = 'Precision_pred'
    if pred_tf:
        combined_df = map_pred_to_tested(combined_df, combined_test, match_cols)
    else:
        combined_df.drop_duplicates(subset=['Model_param_id'], inplace=True)
        sort_col = 'F1_train'

    combined_df.sort_values(sort_col, ascending=False, inplace=True)
    if 'Unnamed: 0' in combined_df.columns:
        combined_df.drop(columns=['Unnamed: 0'], inplace=True)
    combined_df.to_excel(file_path, index=False)
    print(f"XGBoost parameters saved: {file_path}")

    return combined_df


def map_pred_to_tested(pred_df, test_df, match_cols):
    test_df = test_df.drop_duplicates(subset=match_cols)
    merged_df = pred_df.merge(test_df[match_cols + ['Precision_test']], on=match_cols, how='left')
    if 'Precision_test_x' in merged_df.columns:
        merged_df['Precision_test'] = merged_df['Precision_test_x'].combine_first(merged_df['Precision_test_y'])
    return merged_df


def main():
    setup_dict = {
        'side': 'Bull',
        'n_clusters': 12,
        'main_folder': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years\xgboost',
        'data_cols': ['max_depth', 'subsample', 'colsample_bytree', 'learning_rate',
                      'reg_alpha', 'reg_lambda', 'n_estimators']}
    use_given_targets = False
    data_folder = os.path.join(setup_dict['main_folder'], setup_dict['side'], 'data')
    files = get_excel_files(data_folder)

    pred_dfs = []
    test_dfs = []
    for file in files:
        df = pd.read_excel(file)
        if 'tgt_param' not in df.columns:
            df['tgt_param'] = 0
            df.to_excel(file)
        print(file)
        match = re.search(r"(\d+)_xgb_best_params\.xlsx", file)
        if match and len(df) > setup_dict['n_clusters']:
            paramset_id = int(match.group(1))
            param_data = ParamsetResults(setup_dict,
                                         param_df=df,
                                         paramset_id=paramset_id,
                                         file=file,
                                         use_tgt_param=use_given_targets)
        else:
            continue

        param_data.build_clusters()
        param_data.xgboost_model()
        param_data.create_cluster_summary()
        param_data.plot_cluster_data()
        param_data.save_dfs()
        pred_dfs.append(param_data.param_test_df)
        test_dfs.append(param_data.param_df)

    combined_test = pd.concat(test_dfs, ignore_index=True)
    combined_test['is_tgt'], combined_test['tgt_param'] = False, False
    output_folder = os.path.join(setup_dict['main_folder'], setup_dict['side'], 'Data', 'all')
    os.makedirs(output_folder, exist_ok=True)
    output_file = f'{output_folder}\\test_xgb_all_params.xlsx'
    combined_test.to_excel(output_file, index=False)

    final_params = ParamsetResults(setup_dict,
                                   param_df=combined_test,
                                   paramset_id='all',
                                   file=output_file,
                                   use_tgt_param=True)

    final_params.build_clusters()
    final_params.plot_cluster_data()

    combined_pred = pd.concat(pred_dfs, ignore_index=True)
    combined_pred['is_tgt'], combined_pred['tgt_param'] = False, False
    output_file = f'{output_folder}\\pred_xgb_all_params.xlsx'

    "add a string as an argument to identify test vs pred"
    # final_params = ParamsetResults(setup_dict,
    #                                param_df=combined_df,
    #                                paramset_id='all',
    #                                file=output_file,
    #                                use_tgt_param=True)

    save_all_params(combined_pred, output_file,
                    pred_tf=True, combined_test=combined_test)


if __name__ == '__main__':
    main()