import os
import glob
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import hashlib
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
        self.summary_df = pd.DataFrame
        self.data_cols = setup_dict['data_cols']
        self.n_clusters = setup_dict['n_clusters']
        self.tgt_ranges = None
        self.tgt_cluster = int
        self.use_tgt_param = use_tgt_param

    def build_clusters(self):
        self.param_df['paramset_id'] = self.paramset_id
        df = self.param_df[self.param_df['Precision'] != '']
        df_tgt = None

        if self.use_tgt_param:
            df, df_tgt = separate_given_tgt_dfs(df)
        elif self.tgt_ranges:
            df, df_tgt = separate_tgt_dfs(df, self.tgt_ranges)

        df = kmeans_work(df, self.data_cols, self.n_clusters)

        if len(df_tgt) > 0:
            df = pd.concat([df, df_tgt])



        self.param_df = df

    def create_cluster_summary(self):
        work_cols = ['AUC', 'Precision', 'Time'] + self.data_cols
        cluster_summary = self.param_df.groupby('cluster')[work_cols].agg(['mean', 'std', 'min', 'max', 'count'])
        self.summary_df = cluster_summary.sort_values(('Precision', 'mean'), ascending=False)
        self.tgt_cluster = self.summary_df.index[0]

    def create_tgt_ranges(self, use_given=True):
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

        if use_given:
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

    def grow_tgt_cluster(self):
        if not self.tgt_ranges:
            self.create_tgt_ranges()

        tgt_cluster_mask = np.ones_like(self.param_df['cluster'].values)

        for col in self.data_cols:
            col_mask = ((self.tgt_ranges[col][0] <= self.param_df[col]) &
                        (self.param_df[col] <= self.tgt_ranges[col][-1]))
            tgt_cluster_mask = col_mask * tgt_cluster_mask

        self.param_df.loc[tgt_cluster_mask, 'cluster'] = self.tgt_cluster

    def plot_cluster_data(self):
        # if self.use_tgt_param:
        #     self.param_df['is_tgt'] = self.param_df['cluster'] == 'tgt'
        # else:
        self.param_df['is_tgt'] = self.param_df['cluster'] == self.tgt_cluster
        self.param_df.reset_index(drop=True, inplace=True)
        self.param_df['cluster'] = self.param_df['cluster'].astype(str)

        for col in self.data_cols:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=self.param_df, x=col, y='AUC',
                hue='cluster', palette='tab10',
                style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
            )
            sns.regplot(data=self.param_df, x=col, y='AUC', scatter=False, color='darkred', order=3)
            plt.title(f'KMeans AUC for {col}')
            plt.savefig(f'{self.file_folder}\\{self.paramset_id}_{col}_kmeans_AUC.png')
            plt.close()

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

            # Scatterplot for Precision
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                data=self.param_df, x=col, y='Precision',
                hue='cluster', palette='tab10',
                style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
            )
            sns.regplot(data=self.param_df, x=col, y='Precision', scatter=False, color='darkred', order=3)
            plt.title(f'KMeans Precision for {col}')
            plt.savefig(f'{self.file_folder}\\{self.paramset_id}_{col}_kmeans_Precision.png')
            plt.close()

        # AUC vs Precision
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.param_df, x='AUC', y='Precision',
            hue='cluster', palette='tab10',
            style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
        )
        sns.regplot(data=self.param_df, x='AUC', y='Precision', scatter=False, color='darkred', order=3)
        plt.title(f'KMeans AUC vs. Precision')
        plt.savefig(f'{self.file_folder}\\{self.paramset_id}_AUCPrec.png')
        plt.close()

        # Prec vs. Cluster
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=self.param_df, x='cluster', y='Precision',
            hue='cluster', palette='tab10',
            style='is_tgt', markers={True: 'X', False: 'o'}, legend='brief'
        )
        plt.title(f'KMeans cluster vs. Precision')
        plt.savefig(f'{self.file_folder}\\{self.paramset_id}_clusterPrec.png')
        plt.close()

    def save_dfs(self):
        self.summary_df.to_excel(f'{self.file_folder}\\{self.paramset_id}_summary_stats.xlsx')


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
    tgt_mask = (df['tgt_param'] == 1)

    df_tgt = df[tgt_mask].reset_index(drop=True)
    df_tgt['cluster'] = 'tgt'
    df_tgt = assign_ids(df_tgt)
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


def generate_unique_id(row, existing_ids):
    """
    Generate a unique ID for a row based on `paramset_id` and XGBoost params.
    """
    unique_string = f"Algo_{row['paramset_id']}_{row['max_depth']}_{row['learning_rate']}_{row['subsample']}_{row['colsample_bytree']}_{row['reg_alpha']}_{row['reg_lambda']}_{row['n_estimators']}"

    hashed_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]
    full_id = f"Algo_{row['paramset_id']}_{hashed_id}"

    if full_id in existing_ids:
        raise ValueError(f"Duplicate ID detected: {full_id}")
    return full_id


def assign_ids(df):
    """
    Assign unique IDs to rows in a DataFrame based on `paramset_id` and XGBoost params.
    """
    if 'xgb_model_id' in df.columns:
        existing_ids = set(df['xgb_model_id'].dropna().unique())
    else:
        existing_ids = set()

    df['unique_id'] = df.apply(lambda row: generate_unique_id(row, existing_ids), axis=1)

    return df


def main():
    setup_dict = {
        'side': 'Bull',
        'n_clusters': 16,
        'main_folder': r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years\xgboost',
        'data_cols': ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree',
                      'reg_alpha', 'reg_lambda', 'n_estimators']}
    data_folder = os.path.join(setup_dict['main_folder'], setup_dict['side'], 'data')
    files = get_excel_files(data_folder)

    dfs = []
    for file in files:
        df = pd.read_excel(file)
        print(file)
        match = re.search(r"(\d+)_xgb_best_params\.xlsx", file)
        if match and len(df) > setup_dict['n_clusters']:
            paramset_id = int(match.group(1))
            param_data = ParamsetResults(setup_dict,
                                         param_df=df,
                                         paramset_id=paramset_id,
                                         file=file,
                                         use_tgt_param=True)
        else:
            continue

        param_data.build_clusters()
        param_data.create_cluster_summary()
        param_data.grow_tgt_cluster()
        param_data.plot_cluster_data()
        param_data.save_dfs()
        dfs.append(param_data.param_df)

    combined_df = pd.concat(dfs, ignore_index=True)

    output_folder = os.path.join(setup_dict['main_folder'], setup_dict['side'], 'Data', 'all')
    os.makedirs(output_folder, exist_ok=True)
    output_file = f'{output_folder}\\combined_xgb_all_params.xlsx'

    final_params = ParamsetResults(setup_dict,
                                   param_df=combined_df,
                                   paramset_id='all',
                                   file=output_file,
                                   use_tgt_param=True)

    combined_df.to_excel(output_file, index=False)

    final_params.build_clusters()
    final_params.plot_cluster_data()


if __name__ == '__main__':
    main()