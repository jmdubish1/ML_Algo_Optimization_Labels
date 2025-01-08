import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from typing import TYPE_CHECKING
import math
import random
from sklearn.utils.class_weight import compute_class_weight
import cupy as cp
import xgboost as xgb

if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler


class BufferedBatchGenerator(Sequence):
    def __init__(self, process_handler,
                 buffer_size=50,
                 train=True,
                 randomize=False,
                 dat_type='tensor'):
        self.ph = process_handler
        self.batch_size = self.ph.ml_model.batch_s
        self.buffer_size = buffer_size
        self.train_tf = train
        self.sample_ind_list = []
        self.n_samples = 0
        self._current_index = 0
        self.set_attributes(randomize)
        self.dat_type = dat_type

        self.xy_intraday = (self.ph.ml_data.xy_train_intra if self.train_tf
                            else self.ph.ml_data.xy_test_intra).to_numpy()
        self.x_daily = (self.ph.ml_data.x_train_daily if self.train_tf
                        else self.ph.ml_data.x_test_daily).to_numpy()

    def set_attributes(self, randomize=False):
        ncols = self.ph.setup_params.num_y_cols
        hot_enc = (self.ph.ml_data.xy_train_intra if self.train_tf
                   else self.ph.ml_data.xy_test_intra).iloc[:, -ncols:]
        self.sample_ind_list = hot_enc[hot_enc.any(axis=1)].index.tolist()

        if randomize and self.train_tf:
            random.shuffle(self.sample_ind_list)

        self.n_samples = len(self.sample_ind_list)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self._current_index >= self.__len__():
            raise StopIteration

        batch = self.__getitem__(self._current_index)
        self._current_index += 1
        return batch

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        buffer_inds = self.sample_ind_list[start:end]

        if self.dat_type == 'tensor':
            return self._process_buffer_tensor(buffer_inds)
        else:
            return self.process_buffer_numpy(buffer_inds)

    def reset(self):
        self._current_index = 0

    def on_epoch_end(self):
        """Shuffle the data at the end of each epoch if training."""
        if self.train_tf:
            random.shuffle(self.sample_ind_list)

    def _process_buffer_tensor(self, buffer_inds):
        num_y_cols = self.ph.setup_params.num_y_cols
        daily_len = self.ph.ml_model.daily_len
        intra_len = self.ph.ml_model.intra_len
        daily_shape = self.ph.ml_model.input_shape[0]
        intra_shape = self.ph.ml_model.input_shape[1]

        x_day_buffer, x_intra_buffer, y_buffer = [], [], []
        for t_ind in buffer_inds:
            trade_dt = self.xy_intraday[t_ind, 0]

            x_daily_input = self.x_daily[self.x_daily[:, 0] < trade_dt][-daily_len:, 1:]
            x_intra_input = self.xy_intraday[t_ind - intra_len:t_ind, 1:-num_y_cols]
            y_input = self.xy_intraday[t_ind, -num_y_cols:]

            x_day_buffer.append(x_daily_input)
            x_intra_buffer.append(x_intra_input)
            y_buffer.append(y_input)

        x_day_tensor = tf.convert_to_tensor(x_day_buffer, dtype=tf.float32)
        x_intra_tensor = tf.convert_to_tensor(x_intra_buffer, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y_buffer, dtype=tf.float32)

        x_day_tensor = tf.reshape(x_day_tensor, [len(buffer_inds), daily_len, daily_shape[2]])
        x_intra_tensor = tf.reshape(x_intra_tensor, [len(buffer_inds), intra_len, intra_shape[2]])

        return (x_day_tensor, x_intra_tensor), y_tensor

    def process_buffer_numpy(self, buffer_inds):
        num_y_cols = self.ph.setup_params.num_y_cols
        daily_len = self.ph.ml_model.daily_len
        intra_len = self.ph.ml_model.intra_len

        x_day_buffer, x_intra_buffer, y_buffer = [], [], []
        for t_ind in buffer_inds:
            trade_dt = self.xy_intraday[t_ind, 0]

            x_daily_input = self.x_daily[self.x_daily[:, 0] < trade_dt][-daily_len:, 1:]
            x_intra_input = self.xy_intraday[t_ind - intra_len:t_ind, 1:-num_y_cols]
            y_input = self.xy_intraday[t_ind, -num_y_cols:]

            x_day_buffer.append(x_daily_input)
            x_intra_buffer.append(x_intra_input)
            y_buffer.append(y_input)

        x_day_buffer = np.array(x_day_buffer)
        x_intra_buffer = np.array(x_intra_buffer)
        y_buffer = np.array(y_buffer)

        x_buffer = flatten_and_combine(x_day_buffer, x_intra_buffer)
        y_buffer = np.argmax(y_buffer, axis=-1)

        x_buffer = cp.array(x_buffer, dtype=cp.float32)
        y_buffer = cp.array(y_buffer, dtype=cp.float32)

        return x_buffer, y_buffer

    def generate_tf_dataset(self):
        daily_shape = self.ph.ml_model.input_shape[0]
        intra_shape = self.ph.ml_model.input_shape[1]

        t_dataset = tf.data.Dataset.from_generator(
            lambda: iter(self),
            output_signature=(
                (
                    tf.TensorSpec(shape=(None, daily_shape[0], daily_shape[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, intra_shape[0], intra_shape[1]), dtype=tf.float32)
                ),
                tf.TensorSpec(shape=(None, self.ph.setup_params.num_y_cols), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)

        return t_dataset

    def load_full_tensor_dataset(self):
        """
        Load the entire dataset into memory and return it as tensors.
        """
        if self.dat_type == 'tensor':
            (x_day_full, x_intra_full), y_full = self._process_buffer_tensor(self.sample_ind_list)

            full_dataset = (
                tf.data.Dataset.from_tensor_slices(((x_day_full, x_intra_full),
                                                    y_full)).batch(self.batch_size).prefetch(
                    tf.data.AUTOTUNE))
        else:
            (x_day_full, x_intra_full), y_full = self.process_buffer_numpy(self.sample_ind_list)

            full_dataset = (x_day_full, x_intra_full), y_full

        return full_dataset


def load_full_cupy_dataset(data_gen):
    """
    Use the generator to load the entire dataset into CuPy arrays.
    """
    x_list, y_list = [], []

    while True:
        try:
            x_batch, y_batch = next(data_gen)
            x_list.append(x_batch)
            y_list.append(y_batch)
        except StopIteration:
            break

    x_full = cp.vstack(x_list)
    y_full = cp.hstack(y_list)

    return x_full, y_full


def get_class_weights(process_handler):
    ph = process_handler
    y_labels = ph.ml_data.y_train_df['Label']
    uniq_y = np.unique(y_labels)

    y_labels_trade_data = ph.trade_data.y_train_df['Label']

    label_to_index = {label: idx for idx, label in enumerate(uniq_y)}
    numeric_labels_trade = y_labels_trade_data.map(label_to_index)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(label_to_index)),
        y=numeric_labels_trade)

    class_weight_dict = {idx: weight for idx, weight in enumerate(class_weights)}

    class_weight_print = {label: weight for label, weight, in zip(label_to_index.keys(), class_weights)}

    for key, val in class_weight_dict.items():
        if ph.setup_params.over_sample:
            new_val = (val - 1) / 10 + 1
        else:
            new_val = val
        if val == class_weight_print['upper_exit']:
            class_weight_dict[key] = new_val
            class_weight_print['upper_exit'] = new_val
        elif val == class_weight_print['lower_exit']:
            class_weight_dict[key] = new_val
            class_weight_print['lower_exit'] = new_val * 1.05
        elif val == class_weight_print['time_exit']:
            class_weight_dict[key] = new_val
            class_weight_print['time_exit'] = new_val * .975

    class_weight_print = {label: weight for label, weight, in zip(label_to_index.keys(), class_weights)}

    print(f'Class Weights: {class_weight_print}\n')
    print(f'Class Weight 1H: {class_weight_dict}')
    class_weights = np.array(class_weights)

    return class_weight_dict, class_weights


def one_cycle_lr(initial_lr, total_epochs):
    """
    Returns a callable One Cycle Learning Rate Scheduler.

    Parameters:
    - initial_lr (float): Peak learning rate.
    - total_epochs (int): Total number of epochs.

    Returns:
    - A function that computes the learning rate for each epoch.
    """
    max_lr = initial_lr  # Peak learning rate
    min_lr = .0000005  # Minimum learning rate

    def schedule(epoch, lr):
        if epoch < total_epochs // 3:
            # Increase learning rate linearly to the peak
            return min_lr + (max_lr - min_lr) * (epoch / (total_epochs // 3))
        else:
            # Decrease learning rate following a cosine decay
            return min_lr + (max_lr - min_lr) * \
                   (1 + math.cos(math.pi * (epoch - total_epochs // 3) / (total_epochs // 3))) / 3

    return schedule


def flatten_and_combine(x_day_buffer, x_intra_buffer):
    x_day_flat = x_day_buffer.reshape(x_day_buffer.shape[0], -1)
    x_intra_flat = x_intra_buffer.reshape(x_intra_buffer.shape[0], -1)
    x_combined = np.hstack((x_day_flat, x_intra_flat))

    return x_combined
