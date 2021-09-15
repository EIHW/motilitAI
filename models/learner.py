import tensorflow as tf
import pandas as pd
import numpy as np
import math
from abc import ABC, abstractmethod
from hparams import *
from os.path import join
from os import makedirs
from utils import reset_keras
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate, train_test_split
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)



class BaseLearner(ABC):
    def __init__(self, model_dir='model', hparams=None):
        self.best_epoch = 0
        self.model_dir = model_dir
        self.hparams = hparams
        self.val_mae = None
        self.val_rmse = None

    @abstractmethod
    def fit(self, train_X, train_y):
        pass

    @abstractmethod
    def predict(self, test_X):
        pass


class SVRLearner(BaseLearner):
    def __init__(self, model_dir='model', hparams=None, cv=5):
        super().__init__(model_dir=model_dir, hparams=hparams)
        self.model = None
        self.best_epoch = -1
        self.cv = cv

    def fit(self, train_X, train_y, verbose=0):
        self.model = MultiOutputRegressor(SVR(C=self.hparams[HP_COST.name]),
                                          n_jobs=-1)
        scores = cross_validate(
            self.model,
            train_X,
            train_y,
            scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error'],
            cv=self.cv)
        self.val_mae = -sum(scores['test_neg_mean_absolute_error']) / self.cv
        self.val_rmse = -sum(
            scores['test_neg_root_mean_squared_error']) / self.cv
        self.model.fit(train_X, train_y)

    def predict(self, test_X):
        return self.model.predict(test_X)

    def reset(self):
        pass


class KerasLearner(BaseLearner):
    def __init__(self, model_dir='model', hparams=None):
        super().__init__(model_dir=model_dir, hparams=hparams)
        self.model = None
        self.input_shape = None
        self.best_epoch = None
        self.history = None

    @abstractmethod
    def build_model(self, input_shape, output_shape):
        pass

    def fit(self, train_X, train_y, verbose=2):
        makedirs(self.model_dir, exist_ok=True)
        self.build_model(train_X.shape[-1], train_y.shape[1])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=100,
            restore_best_weights=True,
            mode='min')
        X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=42) 
        history = self.model.fit(X_train,
                                 y_train,
                                 batch_size=self.hparams[HP_BATCHSIZE.name],
                                 shuffle=True,
                                 epochs=self.hparams[HP_NUM_EPOCHS.name],
                                 validation_data=(X_val, y_val),
                                 verbose=verbose,
                                 callbacks=[early_stop])

        self.history = pd.DataFrame(history.history)
        best_epoch = np.argmin(history.history['val_mae']) + 1
        if verbose > 0:
            print(self.history, best_epoch)
        best_epoch_vals = self.history.iloc[best_epoch - 1]
        self.val_mae = best_epoch_vals['val_mae']
        self.val_rmse = math.sqrt(best_epoch_vals['val_mse'])
        self.best_epoch = best_epoch
        self.history.to_csv(join(self.model_dir, 'history.csv'))
        self.model.save(join(self.model_dir, 'model.h5'))

    def predict(self, test_X):
        return self.model.predict(test_X)

    def reset(self):
        reset_keras(self.model)


class MLPLearner(KerasLearner):
    def build_model(self, input_shape, output_shape):
        model = tf.keras.Sequential()
        for i in range(self.hparams[HP_NUM_LAYERS.name]):
            if i == 0:
                model.add(
                    tf.keras.layers.Dense(
                        self.hparams[HP_NUM_UNITS.name],
                        input_shape=[input_shape],
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.hparams[HP_REGULARIZER.name])))
            else:
                model.add(
                    tf.keras.layers.Dense(
                        self.hparams[HP_NUM_UNITS.name],
                        kernel_regularizer=tf.keras.regularizers.l2(
                            self.hparams[HP_REGULARIZER.name])))
            model.add(tf.keras.layers.BatchNormalization(renorm=True))
            model.add(
                tf.keras.layers.Activation(self.hparams[HP_ACTIVATION.name]))
            model.add(tf.keras.layers.Dropout(self.hparams[HP_DROPOUT.name]))

        model.add(
            tf.keras.layers.Dense(output_shape,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      self.hparams[HP_REGULARIZER.name])))
        optimizer = tf.keras.optimizers.Adam(
            self.hparams[HP_LEARNING_RATE.name])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model = model


class RNNLearner(KerasLearner):
    def build_model(self, input_shape, output_shape):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Masking(mask_value=-1.0,
                                    input_shape=(None, input_shape)))
        for i in range(self.hparams[HP_NUM_LAYERS.name]):
            return_sequences = i < (self.hparams[HP_NUM_LAYERS.name] - 1)

            if self.hparams[HP_CELL_TYPE.name] == 'gru':
                cell = tf.keras.layers.GRU(
                    self.hparams[HP_NUM_RECURRENT_UNITS.name],
                    dropout=self.hparams[HP_DROPOUT.name],
                    recurrent_dropout=self.hparams[HP_RECURRENT_DROPOUT.name],
                    return_sequences=return_sequences,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hparams[HP_REGULARIZER.name]),
                    recurrent_regularizer=tf.keras.regularizers.l2(
                        self.hparams[HP_REGULARIZER.name]))

            elif self.hparams[HP_CELL_TYPE.name] == 'lstm':
                cell = tf.keras.layers.LSTM(
                    self.hparams[HP_NUM_RECURRENT_UNITS.name],
                    dropout=self.hparams[HP_DROPOUT.name],
                    recurrent_dropout=self.hparams[HP_RECURRENT_DROPOUT.name],
                    return_sequences=return_sequences,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hparams[HP_REGULARIZER.name]),
                    recurrent_regularizer=tf.keras.regularizers.l2(
                        self.hparams[HP_REGULARIZER.name]))

            if self.hparams[HP_BIDIRECTIONAL.name]:
                cell = tf.keras.layers.Bidirectional(cell)
            model.add(cell)
            if return_sequences:
                model.add(tf.keras.layers.LayerNormalization())
        model.add(
            tf.keras.layers.Dense(output_shape,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      self.hparams[HP_REGULARIZER.name])))
        optimizer = tf.keras.optimizers.RMSprop(
            self.hparams[HP_LEARNING_RATE.name])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model = model


class CNNLearner(KerasLearner):
    def build_model(self, input_shape, output_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape(target_shape=(input_shape, 1),
                                    input_shape=[input_shape])
        ])
        for i in range(self.hparams[HP_NUM_LAYERS.name]):
            model.add(
                tf.keras.layers.Convolution1D(
                    2**i * self.hparams[HP_NUM_FILTERS.name],
                    self.hparams[HP_KERNEL_SIZE.name],
                    padding='same',
                    dilation_rate=self.hparams[HP_DILATION_RATE.name],
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2(
                        self.hparams[HP_REGULARIZER.name])))
            model.add(tf.keras.layers.BatchNormalization(renorm=True))
            model.add(
                tf.keras.layers.Activation(self.hparams[HP_ACTIVATION.name]))
            model.add(tf.keras.layers.MaxPooling1D(2))
            model.add(tf.keras.layers.Dropout(self.hparams[HP_DROPOUT.name]))
        model.add(tf.keras.layers.Flatten())
        model.add(
            tf.keras.layers.Dense(output_shape,
                                  kernel_regularizer=tf.keras.regularizers.l2(
                                      self.hparams[HP_REGULARIZER.name])))

        optimizer = tf.keras.optimizers.Adam(
            self.hparams[HP_LEARNING_RATE.name])
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model = model
