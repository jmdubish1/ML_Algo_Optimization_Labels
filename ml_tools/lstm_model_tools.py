import numpy as np
import pandas as pd
import io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2, L1L2
from tensorflow.keras.initializers import GlorotUniform
from ml_tools.custom_callbacks_layers import LivePlotLossesLabels, StopAtAccuracy, MDNLayer
import ml_tools.general_ml_tools as glt
import ml_tools.loss_functions as lf


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ml_tools.process_handler import ProcessHandler

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Check device placement for operations
# tf.debugging.set_log_device_placement(True)


class LstmModel:
    def __init__(self,
                 process_handler: "ProcessHandler",
                 lstm_dict: dict):
        self.ph = process_handler
        self.ph.ml_model = self
        self.lstm_dict = lstm_dict

        self.temperature = lstm_dict['temperature'][self.ph.side]
        self.epochs = self.lstm_dict['epochs'][self.ph.side]
        self.batch_s = lstm_dict['batch_size']
        self.learning_rate = self.lstm_dict['adam_optimizer']
        self.max_acc = lstm_dict['max_accuracy']
        self.intra_len = lstm_dict['intra_lookback']
        self.daily_len = lstm_dict['daily_lookback']
        self.opt_threshold = self.lstm_dict['opt_threshold'][self.ph.side]
        self.buffer = self.lstm_dict['buffer_batch_num']

        self.input_shape = None
        self.input_layer_daily = None
        self.input_layer_intraday = None
        self.win_loss_output = None
        self.float_output = None

        self.model = None
        self.scheduler = None
        self.optimizer = None

        self.model_plot = None
        self.model_summary = None

    def build_compile_model(self):
        print(f'\nBuilding New Model\n'
              f'Param ID: {self.ph.paramset_id}')
        self.build_lstm_model()
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("GPU Details: ", tf.config.list_physical_devices('GPU'))

        self.compile_model()
        self.model.summary()

    def get_input_shapes(self):
        daily_shape = (self.daily_len, self.ph.ml_data.x_train_daily.shape[1] - 1)
        intraday_shape = \
            (self.intra_len, self.ph.ml_data.x_train_intra.shape[1] - 1)

        self.input_shape = (daily_shape, intraday_shape)

    def check_for_retrain(self):
        if self.ph.load_current_model and self.ph.retraintf:
            self.epochs = int(self.epochs/2)
            self.learning_rate = self.learning_rate/2

    def train_model(self, randomize_tf=False):
        self.check_for_retrain()
        epochs = self.epochs
        acc_threshold = self.max_acc

        self.model_plot = LivePlotLossesLabels(plot_live=self.lstm_dict['plot_live'])
        train_gen = glt.BufferedBatchGenerator(self.ph, self.buffer, train=True, randomize=randomize_tf)
        train_gen = train_gen.generate_tf_dataset()
        test_gen = glt.BufferedBatchGenerator(self.ph, self.buffer, train=False)
        test_gen = test_gen.load_full_tensor_dataset()

        stop_at_accuracy = StopAtAccuracy(accuracy_threshold=acc_threshold)
        class_weights, _ = glt.get_class_weights(self.ph)
        one_cyc_lr_fn = glt.one_cycle_lr(self.learning_rate, self.epochs)
        one_cyc_lr = tf.keras.callbacks.LearningRateScheduler(one_cyc_lr_fn)
        self.model.fit(train_gen,
                       epochs=epochs,
                       verbose=1,
                       validation_data=test_gen,
                       callbacks=[one_cyc_lr, self.model_plot, stop_at_accuracy],
                       batch_size=self.batch_s,
                       class_weight=class_weights)
        self.model_plot.save_plot(self.ph.save_handler.data_folder, self.ph.paramset_id)

    def compile_model(self):

        self.optimizer = Adam(self.learning_rate, clipnorm=1.0)
        # threshold = self.opt_threshold

        self.model = Model(inputs=[self.input_layer_daily, self.input_layer_intraday],
                           outputs=self.win_loss_output)

        _, class_weights = glt.get_class_weights(self.ph)
        prec_recall_loss = lf.weighted_prec_recall_loss(class_weights=class_weights)
        metric_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        f1_metric = lf.weighted_f1_loss(metric_weights)
        self.model.compile(optimizer=self.optimizer,
                           loss=prec_recall_loss,
                           metrics=[f1_metric,
                                    prec_recall_loss])

        print('New Model Created')
        self.get_model_summary_df()

    def build_lstm_model(self):
        self.get_input_shapes()
        self.input_layer_daily = Input(self.input_shape[0],
                                       name='daily_input_layer')

        lstm_d1 = Bidirectional(LSTM(units=96,
                                     activation='tanh',
                                     recurrent_activation='sigmoid',
                                     return_sequences=True,
                                     kernel_initializer=GlorotUniform(),
                                     # kernel_regularizer=L1L2(l1=.00075, l2=.005),
                                     kernel_regularizer=l2(.0075),
                                     name='lstm_d1'))(self.input_layer_daily)

        drop_d1 = Dropout(0.05, name='drop_d1')(lstm_d1)

        lstm_d2 = LSTM(units=64,
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(.0075),
                       name='lstm_d2')(drop_d1)

        self.input_layer_intraday = Input(self.input_shape[1],
                                          name='intra_input_layer')

        lstm_i1 = Bidirectional(LSTM(units=self.lstm_dict['lstm_i1_nodes'],
                                     activation='tanh',
                                     recurrent_activation='sigmoid',
                                     return_sequences=True,
                                     kernel_initializer=GlorotUniform(),
                                     # kernel_regularizer=L1L2(l1=.001, l2=.005),
                                     kernel_regularizer=l2(.0075),
                                     name='lstm_i1'))(self.input_layer_intraday)

        drop_i1 = Dropout(0.05, name='drop_i1')(lstm_i1)

        lstm_i2 = LSTM(units=self.lstm_dict['lstm_i2_nodes'],
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       return_sequences=False,
                       kernel_initializer=GlorotUniform(),
                       kernel_regularizer=l2(.0075),
                       name='lstm_i2')(drop_i1)

        merged_lstm = Concatenate(axis=-1,
                                  name='concatenate_timesteps')([lstm_d2, lstm_i2])

        # batch_norm = BatchNormalization(momentum=.75, epsilon=1e-2,
        #                                 name='batch_norm')(merged_lstm)

        dense_m1 = Dense(units=self.lstm_dict['dense_m1_nodes'],
                         activation='sigmoid',
                         kernel_initializer=GlorotUniform(),
                         kernel_regularizer=l2(.005),
                         name='dense_m1')(merged_lstm)

        drop_i1 = Dropout(0.05, name='drop_m1')(dense_m1)

        dense_wl1 = Dense(units=self.lstm_dict['dense_wl1_nodes'],
                          activation='sigmoid',
                          kernel_initializer=GlorotUniform(),
                          kernel_regularizer=l2(.0075),
                          # kernel_regularizer=L1L2(l1=.001, l2=.005),
                          name='dense_wl1')(drop_i1)

        # Output layers
        self.win_loss_output = Dense(self.ph.setup_params.num_y_cols,
                                     activation='softmax',
                                     name='wl_class')(dense_wl1)

    def get_model_summary_df(self, printft=False):
        if printft:
            self.model.summary()

        summary_buf = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))

        summary_string = summary_buf.getvalue()
        summary_lines = summary_string.split("\n")

        summary_data = []
        for line in summary_lines:
            split_line = list(filter(None, line.split(" ")))
            if len(split_line) > 1:
                summary_data.append(split_line)

        df_summary = pd.DataFrame(summary_data)
        df_cols = df_summary.iloc[1]
        df_summary = df_summary.iloc[2:].reset_index(drop=True)
        df_summary.columns = df_cols

        self.model_summary = df_summary





