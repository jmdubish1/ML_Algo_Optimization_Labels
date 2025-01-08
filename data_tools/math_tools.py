import pandas as pd
import numpy as np
from numba import jit
from arch import arch_model
from itertools import combinations


def create_atr(df, sec, n=8):
    high_low = df[f'{sec}_High'] - df[f'{sec}_Low']
    high_prev_close = np.abs(df[f'{sec}_High'] - df[f'{sec}_Close'].shift(1))
    low_prev_close = np.abs(df[f'{sec}_Low'] - df[f'{sec}_Close'].shift(1))
    true_range = np.maximum(high_low, high_prev_close)
    true_range = np.maximum(true_range, low_prev_close)

    # Calculate Average True Range (ATR)
    atr = np.zeros_like(df[f'{sec}_Close'])
    atr[n - 1] = np.mean(true_range[:n])  # Initial ATR calculation

    for i in range(n, len(df[f'{sec}_Close'])):
        atr[i] = ((atr[i - 1] * (n - 1)) + true_range[i]) / n

    return atr


def create_rsi(series, period=14):
    # Calculate price differences
    delta = series.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Compute exponential moving averages of gains and losses
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_smooth_rsi(series, period=14, smooth=9):
    rsi = create_rsi(series, period)
    rsi_temp = pd.Series(rsi)
    rsi_smooth = rsi_temp.rolling(window=smooth).mean()

    return rsi, rsi_smooth


def add_high_low_diff(df, sec):
    df[f'{sec}_HL_diff'] = (
            df[f'{sec}_High'] - df[f'{sec}_Low']) / ((df[f'{sec}_High'] + df[f'{sec}_Low'])/2)*1000
    df[f'{sec}_OC_diff'] = (
            df[f'{sec}_Open'] - df[f'{sec}_Close']) / ((df[f'{sec}_Open'] + df[f'{sec}_Close'])/2)*1000

    df[f'{sec}_HL_Ratio'] = df[f'{sec}_HL_diff'] / df[f'{sec}_HL_diff'].shift(1)
    df[f'{sec}_OC_Ratio'] = df[f'{sec}_OC_diff'] / df[f'{sec}_OC_diff'].shift(1)

    return df


def calculate_ema_numba(df, price_colname, window_size, smoothing_factor=2):
    result = calculate_ema_inner(
        price_array=df[price_colname].to_numpy(),
        window_size=window_size,
        smoothing_factor=smoothing_factor
    )

    return result


@jit(nopython=True)
def calculate_ema_inner(price_array, window_size, smoothing_factor):
    result = np.empty(len(price_array), dtype="float64")
    sma_list = list()
    for i in range(len(result)):

        if i < window_size - 1:
            result[i] = np.nan
            sma_list.append(price_array[i])
        elif i == window_size - 1:
            sma_list.append(price_array[i])
            result[i] = sum(sma_list) / len(sma_list)
        else:
            result[i] = ((price_array[i] * (smoothing_factor / (window_size + 1))) +
                         (result[i - 1] * (1 - (smoothing_factor / (window_size + 1)))))

    return result


def standardize_ema(arr, lag=12):
    arr = np.array(arr)
    standardized_arr = np.ones_like(arr, dtype=np.float32)

    standardized_arr[lag:] = arr[lag:] / arr[:-lag]

    return standardized_arr


def encode_time_features(df: pd.DataFrame,
                         time_frame: str):
    # Cyclic encoding for Month, Day, Hour
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['DayofWeek_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
    df['DayofWeek_cos'] = np.cos(2 * np.pi * df['Day'] / 7)

    if time_frame != 'daily':
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Minute_sin'] = np.cos(2 * np.pi * df['Minute'] / 60)
        df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)

    for col in ['Month', 'Day', 'Hour', 'Minute', 'DayofWeek']:
        if col in df.columns:
            df = df.drop(columns=col, errors='ignore')

    return df


def subset_to_first_nonzero(arr):
    first_nonz_ind = np.argmax(arr != 0)
    trimmed_arr = arr[first_nonz_ind:]

    return trimmed_arr


def garch_modeling(df, sec):
    print(f'Modeling EGARCH')
    for met in ['Close', 'Vol']:
        print(f'...{sec}_{met}')
        temp_df = df[f'{sec}_{met}']
        temp_df = rescale_data_to_range(temp_df, 500)
        garch_m = arch_model(temp_df, vol='GARCH', p=1, q=1)
        garch_fit = garch_m.fit(disp='off')
        df[f'{sec}_{met}_garch_cv'] = garch_fit.conditional_volatility
        df[f'{sec}_{met}_garch_std'] = garch_fit.std_resid

        # garch_m = arch_model(temp_df, vol='EGARCH', p=1, o=1, q=1)
        # garch_fit = garch_m.fit(disp='off')
        # df[f'{sec}_{met}_egarch_cv'] = garch_fit.conditional_volatility
        # df[f'{sec}_{met}_egarch_std'] = garch_fit.std_resid

    return df


def rescale_data_to_range(df, max_range=1000):
    if df.abs().max() > max_range:
        scale_factor = max_range / df.abs().max()
        temp_df_scaled = df * scale_factor
    else:
        temp_df_scaled = df

    return temp_df_scaled


def calculate_max_drawdown(pnl_series):
    draw_list = [0]
    arr = pnl_series.values
    for i in range(1, len(arr)):
        prev_max = np.max(arr[:i])
        draw_list.append(min(0, arr[i] - prev_max, arr[i-1]))

    if len(draw_list) > 1:
        draw_list.pop(0)
        draw_list.append(draw_list[-1])

    return draw_list


def calculate_algo_lstm_ratio(algo_series, lstm_series, max_lever):
    draw_ratio_list = [1]
    algo_arr = algo_series.values
    lstm_arr = lstm_series.values
    for i in range(1, (len(algo_arr))):
        start_ind = max(0, i-50)
        if np.min(lstm_arr[start_ind:i]) == 0:
            draw_ratio_list.append(1)
        else:
            algo_1 = algo_arr[start_ind:i]
            algo_1 = algo_1[algo_1 != 0]

            lstm_1 = lstm_arr[start_ind:i]
            lstm_1 = lstm_1[lstm_1 != 0]

            if (len(lstm_1) == 0) or (len(algo_1) == 0):
                draw_ratio_list.append(draw_ratio_list[-1])
            else:
                max_draw = np.median(algo_1)/np.median(lstm_1)
                draw_ratio_list.append(max(1, min(max_draw, max_lever)))

    return draw_ratio_list


def sortino_ratio(returns, risk_free_rate=0, target_return=0):
    """
    Calculate the Sortino Ratio for a series of returns.

    Parameters:
    - returns (array-like): Array or list of portfolio returns.
    - risk_free_rate (float): The risk-free rate of return (default is 0).
    - target_return (float): The target return (default is 0, same as the risk-free rate).

    Returns:
    - float: The Sortino Ratio.
    """
    returns = np.array(returns)

    excess_return = np.mean(returns) - risk_free_rate

    downside_returns = returns[returns < target_return]
    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2)) if len(
        downside_returns) > 0 else np.nan

    if downside_deviation > 0:
        sortino = excess_return / downside_deviation
    else:
        sortino = np.nan

    return sortino


def find_percentile_for_percent_sum(arr, percentile):
    """
    Find the percentile threshold where the sum of elements greater than
    this threshold equals or exceeds 50% of the total sum of the array.

    Parameters:
    - arr (array-like): Input array of numerical values.

    Returns:
    - percentile (float): The percentile threshold.
    """
    arr = np.array(arr)  # Ensure the input is a numpy array
    total_sum = np.sum(arr)  # Calculate total sum
    half_sum = total_sum * percentile/100  # 50% of the total sum

    # Sort the array in descending order
    sorted_arr = np.sort(arr)[::-1]

    cumulative_sum = 0  # To track running cumulative sum
    threshold = None  # To store the threshold value

    for val in sorted_arr:
        cumulative_sum += val
        if cumulative_sum >= half_sum:
            threshold = val
            break

    # Calculate the percentile of the threshold value
    # percentile = np.percentile(arr, 100 * (np.sum(arr > threshold) / len(arr)))

    return threshold


def clip_array(arr, low_percentile=1, high_percentile=99):
    clip_low = np.percentile(arr, low_percentile)
    clip_high = np.percentile(arr, high_percentile)

    return np.clip(arr, clip_low, clip_high)


def calculate_bollinger_bands(df, sec, length=14, std_dev=2):
    """
    Calculate Bollinger Bands using EMA.

    Parameters:
    - data (pd.Series): Series of stock prices (e.g., Close prices).
    - length (int): Length of the EMA (default: 14).
    - std_dev (int): Number of standard deviations for the bands (default: 2).

    Returns:
    - pd.DataFrame: Original data with 'EMA', 'Upper Band', and 'Lower Band' columns.
    """
    data = df[f'{sec}_Close']
    ema = data.ewm(span=length, adjust=False).mean()

    rolling_std = data.ewm(span=length, adjust=False).std()

    upper_band = (ema + (rolling_std * std_dev))/ema
    lower_band = (ema - (rolling_std * std_dev))/ema

    return upper_band, lower_band


def sec_correlation(df, sec_list):
    temp_sec_list = [f'{sec}_Close' for sec in sec_list]
    sec_combos = list(combinations(temp_sec_list, 2))

    for col1, col2 in sec_combos:
        col_name = f'Roll_corr_{col1}_{col2}'
        df[col_name] = df[col1].rolling(14).corr(df[col2])

    return df

