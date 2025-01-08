import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from fracdiff import fdiff
from fracdiff.sklearn import FracdiffStat
import glob


warnings.filterwarnings(action="ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
pd.set_option('display.max_rows', None, 'display.max_columns', None)


def adj_columns(df):
    for col in df.columns[1:]:
        df[col] = df[col].astype(np.float32)

    return df


def adf_d_FFD_matrix(arr, window_size=50, d_list=None):
    if d_list is None:
        d_list = np.linspace(0, 1, 11)
    arr = pd.Series(arr).fillna(method='ffill').dropna().values

    adf_data = []
    test_arrs = []

    last_adf = 1
    adf_crit = 0
    tries = 0
    while last_adf > adf_crit and tries < len(d_list):
        for d in d_list:
            test_arr = fdiff(arr, d, window=window_size, mode='same')
            test_arr = subset_to_first_nonzero(test_arr)
            padding = len(arr) - len(test_arr)
            test_arr = test_arr[~np.isnan(test_arr)]
            test_arr = np.pad(test_arr, (padding, 0), constant_values=0)
            corr = np.corrcoef(arr, test_arr)[0, 1]
            adf = adfuller(test_arr, maxlag=8, autolag='AIC')
            adf = [i for i in adf]
            adf.append(corr)
            adf.append(d)
            adf.append(len(test_arr))
            test_arrs.append([d, test_arr])
            adf_data.append(adf)

            last_adf, adf_crit = adf_data[-1][0], adf_data[-1][4]['1%']
            tries += 1

    return adf_data, test_arrs


def organize_adf_data(adf_data):
    adf_df = pd.DataFrame(adf_data)
    adf_df.columns = ['adf_val', 'pval', 'usedlag', 'nobs', 'critical_vals', 'icbest', 'corr', 'd_val', 'max_len']

    return adf_df


def adjust_for_inflation(df, cpi_df, col):
    df_ = df[['Date', col]]
    cpi_df_ = cpi_df[['observation_date', 'total_inflation']]
    df_['YearMonth'] = df_['Date'].dt.to_period('M')
    cpi_df_['YearMonth'] = cpi_df_['observation_date'].dt.to_period('M')

    df_ = df_.merge(cpi_df_[['YearMonth', 'total_inflation']],
                    on='YearMonth', how='left')
    inflation_arr = df_['total_inflation'].values
    inflation_arr = inflation_arr/inflation_arr[0]
    df['Close'] = df_['Close'] * inflation_arr[::-1]

    return df


def plot_adf_test(adf_data, save_name):
    d = adf_data['d_val']
    conf = adf_data.loc[0, 'critical_vals']['1%']

    adf_data['corr'] = adf_data['corr']
    plot_data = adf_data[['d_val', 'pval', 'adf_val']]

    plt_ = plt
    plt_.figure(figsize=(8, 6))
    fig, ax1 = plt_.subplots()
    ax1.plot(d, adf_data['corr'], label='Corr', color='blue')

    ax2 = ax1.twinx()
    bbox = ax1.get_position()
    ax2.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height])
    ax2.plot(d, adf_data['adf_val'], label='adf-test (right)', color='darkred')

    ax2.axhline(conf, label=f'Confidence Min {conf: .3f}', color='black', linestyle='--')
    ax1.axhline(0.9, label=f'Correlation Min {0.9}', color='blue', linestyle='-.')

    # Add titles and labels
    plt_.title('ADF Test')
    plt_.xlabel('d-val')

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    plot_labels = plot_data.columns
    plot_data = plot_data.round(5).values

    plt_.table(cellText=plot_data,
               colLabels=plot_labels,
               loc='center',
               bbox=[1.2, 0.1, 0.4, 0.8])  # [x, y, width, height]

    plt_.subplots_adjust(right=0.75)

    plt_.grid(True)
    plt_.tight_layout()

    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.savefig(save_name)
    # plt.show()
    plt.close(fig)


def plot_frac_FFD(test_arrs, df, save_loc, plot_name):
    arr = df['Close'].values
    dates = pd.to_datetime(df['Date']).values
    for data_array in test_arrs:
        d = data_array[0]
        w_df = data_array[1]

        valid_idx = len(dates) - len(w_df)
        x = dates[-valid_idx:]
        y1 = arr[-valid_idx:]
        y2 = w_df
        y2_avg = np.nanmean(y2)

        fig, ax1 = plt.subplots()
        ax1.plot(x, y1, 'b-', label='Vol - Real')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Vol - Real', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))

        ax2.plot(x, y2, 'r-', label='Vol - FracDiff')
        ax2.axhline(y2_avg, color='green', linestyle='--', label=f'y2 avg = {y2_avg:.2f}')
        ax2.set_ylabel('Vol - FracDiff', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.legend(loc='upper right')

        plt.title(f'd: {d}')
        plt.tight_layout()
        save_name = f'{save_loc}\\plots\\{plot_name}.png'
        plt.savefig(save_name)
        plt.close(fig)


def plot_best_ffd_params(save_loc, df, arr, sec_, time_f, test_c):
    ffd_df = pd.read_excel(f'{save_loc}\\all_FFD_params.xlsx')

    test_params = ffd_df[(ffd_df['Data'] == test_c) &
                         (ffd_df['security'] == sec_) &
                         (ffd_df['time_frame'] == time_f)]

    d = test_params['d_val'].values[0]
    ws = test_params['window'].values[0]
    w_df = fdiff(arr, d, window=ws, mode='same')

    dates = pd.to_datetime(df['Date']).values
    plot_name = f'{sec_}_{time_f}_{test_c}'
    valid_idx = len(dates) - len(w_df) + 1
    x = dates[valid_idx:]
    y1 = arr[valid_idx:]
    y2 = w_df[1:]
    y2_avg = np.nanmean(y2)

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, 'b-', label='Vol - Real')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Vol - Real', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 60))

    ax2.plot(x, y2, 'r-', label='Vol - FracDiff')
    ax2.axhline(y2_avg, color='green', linestyle='--', label=f'y2 avg = {y2_avg:.2f}')
    ax2.set_ylabel('Vol - FracDiff', color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    ax2.legend(loc='upper right')

    plt.title(f'd: {d: .3f}\n'
              f'window: {ws}')
    plt.tight_layout()
    save_name = f'{save_loc}\\plots\\{plot_name}.png'
    plt.savefig(save_name)
    plt.close(fig)


def build_optimal_d_windows(arr, window_arr, test_col):
    final_ds = []
    for window in window_arr:
        print(window)
        adf_data, _ = adf_d_FFD_matrix(arr, window_size=window)
        adf_data = organize_adf_data(adf_data)

        adf_vals = adf_data['adf_val'].values
        crit_val = adf_data.at[0, 'critical_vals']['1%']
        best_ind = np.argmax(adf_vals < crit_val)

        if best_ind == 0 and test_col == 'Close':
            continue
        elif best_ind > 0:
            best_d_val = adf_data.loc[best_ind - 1, 'd_val']
            new_test_ds = [best_d_val + i / 100 for i in range(11)]
        else:
            new_test_ds = [.01 + i / 100 for i in range(10)]

        adf_data2, _ = adf_d_FFD_matrix(arr, window_size=window, d_list=new_test_ds)
        adf_data2 = organize_adf_data(adf_data2)
        final_d_ind = np.argmax(adf_data2['adf_val'].values < crit_val)
        final_d = adf_data2.iloc[final_d_ind]
        final_d['window'] = window

        final_ds.append(final_d)

    final_ds = pd.concat(final_ds, axis=1).T
    final_ds.reset_index(inplace=True)

    return final_ds


def find_optimal_d(arr, weight_stats, window_arr, test_col):
    final_ds = build_optimal_d_windows(arr, window_arr, test_col)
    final_ds = score_d_corr(final_ds, weight_stats)
    optimal_d_window = final_ds.iloc[np.argmax(final_ds['d_corr_score'].values)]

    return optimal_d_window


def score_d_corr(adf_data, weights_stats):
    w1 = weights_stats['w_corr']
    w2 = weights_stats['w_d']
    scores = []
    for ind, row in adf_data.iterrows():
        corr = row['corr']
        adf_stat = row['adf_val']
        d_val = row['d_val']
        score = ((w1 * (1 - (1 - corr)) + w2 * (1 - d_val)) *
                 (1 if adf_stat <= row['critical_vals']['1%'] else 0) * (1 if d_val > .1 else .75))
        scores.append(score)
    adf_data['d_corr_score'] = scores

    return adf_data


def subset_to_first_nonzero(arr):
    first_nonz_ind = np.argmax(arr != 0)
    trimmed_arr = arr[first_nonz_ind:]

    return trimmed_arr


def load_prep_data(data_loc, data_end):
    remove_cols = ['Vol.1', 'OI', 'Time', 'AvgExp12', 'AvgExp24', 'Bearish_Double_Candle',
                   'Bullish_Double_Candle', 'VolAvg']

    df = pd.read_csv(f'{data_loc}\\{data_end}')
    df_cols = df.columns
    df_cols = [col for col in df_cols if col not in remove_cols]
    df = df[df_cols]
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df[(df['Date'] >= pd.to_datetime('2010-04-01', format='%Y-%m-%d')) &
            (df['Date'] < pd.to_datetime('2022-04-01', format='%Y-%m-%d'))]
    df = adj_columns(df)

    return df


def save_best_params(best_param, sec_, time_f, test_c, save_loc):
    best_param = pd.DataFrame(best_param).T
    best_param['security'] = sec_
    best_param['time_frame'] = time_f
    best_param['Data'] = test_c

    col_start = ['security', 'time_frame', 'Data']
    col_order = col_start + [col for col in best_param.columns if col not in col_start]
    best_param = best_param[col_order]

    bp_save = f'{save_loc}\\all_FFD_params.xlsx'
    if os.path.exists(bp_save):
        temp_df = pd.read_excel(bp_save)
        cond = ((temp_df['security'] == sec_) &
                (temp_df['time_frame'] == time_f) &
                (temp_df['Data'] == test_c))
        if cond.any():
            temp_df.iloc[cond] = best_param.iloc[0]
        else:
            temp_df = pd.concat([temp_df, best_param], ignore_index=True)
        temp_df.to_excel(bp_save, index=False)
    else:
        best_param.to_excel(bp_save, index=False)


def main():
    data_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
    strat_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR'
    save_loc = f'{strat_loc}\\agg_data'
    securities = ['NQ', 'GC', 'CL', 'RTY', 'ES', 'YM']
    time_frames = ['daily', '15min', '5min']

    cpi_file = f'{data_loc}\\inflation_adjusted\\CPIAUCSL_seasonally_adj.xlsx'
    cpi_df = pd.read_excel(cpi_file, sheet_name='Total_inflation')
    cpi_df['observation_date'] = pd.to_datetime(cpi_df['observation_date'])

    weights = {'w_corr': .9,
               'w_d': .1}

    window_dict = {'Close': list(range(25, 126, 25)),
                   'Vol':  list(range(15, 26, 5)),
                   'OpenInt': list(range(15, 26, 5))}

    plot_opt_window = False
    plot_opt_d = False
    optimize_d_window = True

    for sec in securities:
        for timef in time_frames:
            data_end = f'{sec}_{timef}_20240505_20040401.txt'
            df = load_prep_data(data_loc, data_end)

            # threshold = df['OpenInt'].quantile(0.50)
            # df['OpenInt'] = df['OpenInt'].where(df['OpenInt'] < threshold, 250000)
            # # df['OpenInt'].shift(5)
            # df['OpenInt'] = df['OpenInt'].fillna(df['OpenInt'].iloc[0:5])

            for test_col in ['Close', 'Vol']:

                windows = window_dict[test_col]
                print(f'{sec} : {timef}: {test_col}')
                os.makedirs(save_loc, exist_ok=True)

                inflate = False
                log_scale = False

                if inflate:
                    df = adjust_for_inflation(df, cpi_df, test_col)

                arr = df[test_col].values

                # if sec in ['NQ', 'YM', 'ES', 'RTY']:
                #     arr = np.log(arr)

                if optimize_d_window:
                    if sec in ['GC', 'CL']:
                        windows = list(range(11, 21, 2))

                    best_param = find_optimal_d(arr, weight_stats=weights, window_arr=windows, test_col=test_col)
                    save_best_params(best_param, sec, timef, test_col, save_loc)

                if plot_opt_window:
                    for ws in windows:
                        print(f'{ws}')
                        d_list = np.linspace(0, 1, 21)
                        adf_data, test_arrs = adf_d_FFD_matrix(arr, window_size=ws, d_list=d_list)
                        adf_data = organize_adf_data(adf_data)
                        # plot_frac_FFD(test_arrs, df, save_loc)

                        save_name = f'{save_loc}\\plots\\{sec}_{test_col}_d_{ws}_analysis.png'
                        os.makedirs(os.path.dirname(save_name), exist_ok=True)

                        try:
                            plot_adf_test(adf_data, save_name)

                        except:
                            continue

                if plot_opt_d:
                    plot_best_ffd_params(save_loc, df, arr, sec, timef, test_col)



if __name__ == '__main__':
    main()