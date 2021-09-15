from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import math
import argparse
import glob
from tensorboard.plugins.hparams import api as hp
from os.path import join, splitext, basename
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform


PATIENT_RELATED = ['abstinence', 'age']
MOTILITY_LABELS = ['immotile_sperm', 'progressive_motility', 'non_progressive_motility']
MORPHOLOGY_LABELS = ['head_defects', 'mid_defects', 'tail_defects']


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2,3,4]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([200, 300, 500]))
HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([64, 128, 256]))

HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.3, 0.5]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-4, 0.1))

#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([tf.keras.optimizers.Nadam, tf.keras.optimizers.Adam]))

METRIC_MEAN_MAE = 'mean_mae'
METRIC_MEAN_RMSE = 'mean_rmse'

PARAM_GRID = {HP_NUM_UNITS.name: HP_NUM_UNITS.domain.values,
HP_NUM_LAYERS.name: HP_NUM_LAYERS.domain.values,
HP_NUM_EPOCHS.name: HP_NUM_EPOCHS.domain.values,
HP_BATCHSIZE.name: HP_BATCHSIZE.domain.values,
HP_DROPOUT.name: HP_DROPOUT.domain.values,
HP_LEARNING_RATE.name: np.logspace(np.log10(HP_LEARNING_RATE.domain.min_value), np.log10(HP_LEARNING_RATE.domain.max_value), num=4)}





def neural_network(fold_datasets, fold_labels, hparams, remove_labels=MORPHOLOGY_LABELS, use_prd=False,  model_dir='model'):
    train_dataset = pd.concat([pd.read_csv(fold_files[0]), pd.read_csv(fold_files[1])], ignore_index=True)

    dataset = train_dataset.copy()
    dataset.tail()

    #devel_dataset = pd.read_csv(fold_files[1])
    test_dataset = pd.read_csv(fold_files[2])

    train_dataset.pop('id_scene')
    #devel_dataset.pop('id_scene')
    test_dataset.pop('id_scene')

    if not use_prd:
        for p in PATIENT_RELATED:
            train_dataset.pop(p)
            #devel_dataset.pop(p)
            test_dataset.pop(p)

    train_labels = np.stack([train_dataset.pop(fold_label) for fold_label in fold_labels], axis=-1)
    print(train_labels.shape)
    #devel_labels = np.stack([devel_dataset.pop(fold_label) for fold_label in fold_labels], axis=-1)
    test_labels = np.stack([test_dataset.pop(fold_label) for fold_label in fold_labels], axis=-1)


    
    for remove_label in remove_labels:
        train_dataset.pop(remove_label)
        #devel_dataset.pop(remove_label)
        test_dataset.pop(remove_label)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    
   

    train_dataset = norm(train_dataset)
    #devel_dataset = norm(devel_dataset)
    test_dataset = norm(test_dataset)
    # normed_train_data = train_dataset
    # normed_devel_data = devel_dataset
    # normed_test_data = test_dataset
    
    # normed_train_labels = train_labels #norm_labels(train_labels) #
    # normed_devel_labels = devel_labels #norm_labels(devel_labels) #
    # normed_test_labels = test_labels


    def build_cnn(hparams=None):
        model = keras.Sequential([
        layers.Reshape(target_shape=(len(train_dataset.keys()),1), input_shape=[len(train_dataset.keys())]),
        layers.Convolution1D(32, 3, strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        layers.Convolution1D(64, 3, strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.5),
        layers.Convolution1D(128, 3, strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(train_labels.shape[1], #activation='sigmoid'
        )
        ])
        optimizer = tf.keras.optimizers.Nadam(hparams[HP_LEARNING_RATE.name])
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model


    def build_lstm():
        model = keras.Sequential([
        layers.Reshape(target_shape=(len(train_dataset.keys()),1), input_shape=[len(train_dataset.keys())]),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(train_labels.shape[1], #activation='sigmoid'
        )
        ])
        optimizer = tf.keras.optimizers.Adam(1e-3)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model

    def build_mlp(hparams):
        model = keras.Sequential()
        for i in range(hparams[HP_NUM_LAYERS.name]):
            if i == 0:
                model.add(layers.Dense(hparams[HP_NUM_UNITS.name], input_shape=[len(train_dataset.keys())]))
            else:
                model.add(layers.Dense(hparams[HP_NUM_UNITS.name]))
            model.add(layers.Activation('relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(hparams[HP_DROPOUT.name]))
       
        model.add(layers.Dense(train_labels.shape[1]))
        optimizer = tf.keras.optimizers.Nadam(hparams[HP_LEARNING_RATE.name])
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model


    

    model = build_mlp(hparams)
    model.summary()
    
    EPOCHS = hparams[HP_NUM_EPOCHS.name]
    
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=patience, restore_best_weights=True, mode='min')

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=join(model_dir, 'model.h5'), monitor='val_mean_absolute_error', save_best_only=False, period=20, mode='min')

    tb_callback = keras.callbacks.TensorBoard(log_dir=join(model_dir, 'logs'))
    history = model.fit(train_dataset, train_labels,  batch_size=hparams[HP_BATCHSIZE.name], shuffle=True,
                        epochs=EPOCHS, validation_data=(test_dataset, test_labels), verbose=0, 
                        callbacks=[#tfdocs.modeling.EpochDots(), 
                        model_checkpoint, tb_callback])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

    print("Test set Mean Abs Error: {:5.4f}".format(mae))

    # devel_predictions = model.predict(devel_dataset)

    rmse = math.sqrt(mse)
    return mae, rmse

def optimize_parameters(fold_files, fold_labels, n_iter=10, remove_labels=MORPHOLOGY_LABELS, use_prd=True, model_dir='model'):
    with tf.summary.create_file_writer(join(model_dir, 'logs', 'hparam_tuning')).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_NUM_LAYERS, HP_LEARNING_RATE, HP_NUM_EPOCHS, HP_DROPOUT],
            metrics=[hp.Metric(METRIC_MEAN_MAE, display_name='Mean MAE (across folds)'), 
            hp.Metric(METRIC_MEAN_RMSE, display_name='Mean RMSE (across folds)')],
        )
    session_num = 0
    param_list = list(ParameterSampler(PARAM_GRID, n_iter=n_iter, random_state=42))
    for hparams in param_list:
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        print(hparams)
        all_maes, all_rmses = run_cross_validation(fold_files, fold_labels, hparams=hparams, remove_labels=MORPHOLOGY_LABELS, use_prd=True, model_dir=join(model_dir, 'logs', 'hparam_tuning', run_name))
        session_num += 1




def run_cross_validation(fold_files, fold_labels, hparams, remove_labels=MORPHOLOGY_LABELS, use_prd=True, model_dir='model'):
    all_mae = []
    all_rmse = []

    

    # fold_maes = []
    # fold_rmses = []
    for i in range(len(fold_files)):
        print(f'-------------------------------------------- fold {i+1} -----------------------------------')
        folds = list(range(len(fold_files)))
        test_file = fold_files[i]
        folds.pop(i)
        print(folds)

        # for j in range(len(folds)):
        #     remaining_folds = folds.copy()
        train_file = fold_files[folds[0]]
        devel_file = fold_files[folds[1]]
        mae, rmse = neural_network([train_file, devel_file, test_file], fold_labels, hparams=hparams, remove_labels=remove_labels, use_prd=False, model_dir=join(model_dir, f'fold-{i+1}'))
        all_mae.append(mae)
        all_rmse.append(rmse)
        # fold_maes.append(sum(maes)/len(maes))
        # fold_rmses.append(sum(rmses)/len(rmses))
    with tf.summary.create_file_writer(model_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mean_mae = sum(all_mae)/len(all_mae)
        mean_rmse = sum(all_rmse)/len(all_rmse)

        tf.summary.scalar(METRIC_MEAN_MAE, mean_mae, step=1)
        tf.summary.scalar(METRIC_MEAN_RMSE, mean_rmse, step=1)
        print(f'All MAEs: {all_mae} mean: {mean_mae}, all RMSEs: {all_rmse} mean: {mean_rmse}')


    return all_mae, all_rmse

        
    # mae_f123, rmse_f123 = neural_network(fold1, fold2, fold3, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    # mae_f213, rmse_f213 = neural_network(fold2, fold1, fold3, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    # print('-------------------------------------------- fold 2 -----------------------------------')
    # mae_f132, rmse_f132 = neural_network(fold1, fold3, fold2, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    # mae_f312, rmse_f312 = neural_network(fold3, fold1, fold2, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    # print('-------------------------------------------- fold 3 -----------------------------------')
    # mae_f231, rmse_f231 = neural_network(fold2, fold3, fold1, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    # mae_f321, rmse_f321 = neural_network(fold3, fold2, fold1, 'immotile_sperm', 'progressive_motility', 'non_progressive_motility')
    
    # result_mae_fold1 = (mae_f231 + mae_f321)/2
    # result_mae_fold2 = (mae_f132 + mae_f312)/2
    # result_mae_fold3 = (mae_f123 + mae_f213)/2

    # result_rmse_fold1 = (rmse_f231 + rmse_f321)/2
    # result_rmse_fold2 = (rmse_f132 + rmse_f312)/2
    # result_rmse_fold3 = (rmse_f123 + rmse_f213)/2

    # print('Mean Absolute Error (MAE):')
    # print(f'fold1: {round(result_mae_fold1, 4)}')
    # print(f'fold2: {round(result_mae_fold2, 4)}')
    # print(f'fold3: {round(result_mae_fold3, 4)}')
    # print(f'average: {round((result_mae_fold1+result_mae_fold2+result_mae_fold3)/3, 4)}')
    # print('Root Mean Square Error (RMSE):')
    # print(f'fold1: {round(result_rmse_fold1, 4)}')
    # print(f'fold2: {round(result_rmse_fold2, 4)}')
    # print(f'fold3: {round(result_rmse_fold3, 4)}')
    # print(f'average: {round((result_rmse_fold1+result_rmse_fold2+result_rmse_fold3)/3, 4)}')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train and evaluate model on 3-fold emsd cross-validation.')
    # parser.add_argument('feature_files', metavar='X', type=str, nargs=3,
    #                     help='Supply emsd feature files for fold 1, 2 and 3.')
    # parser.add_argument('-ma', '--model-architecture', type=str, choices=['mlp'], default='cnn',
    #                     help='Type of neural network architecture (default: CNN)')
    # parser.add_argument('-md', '--model-dir', type=str,  default='model',
    # help='Model directory.')
    # parser.add_argument('-ri', '--random-iterations', type=int,  default=10,
    # help='Number of random grid serach iterations.')
    # parser.add_argument('-l', '--target-labels', type=str, choices=['motility', 'morphology'], 
    #                     default='motility', help='Target labels (default: motility)')
    # args = parser.parse_args()


    # if args.target_labels == 'motility':
    #     fold_labels = MOTILITY_LABELS
    #     remove_labels = MORPHOLOGY_LABELS
    # elif args.target_labels == 'morphology':
    #     fold_labels = MORPHOLOGY_LABELS
    #     remove_labels = MOTILITY_LABELS



    # not-overlapping:
    # fold1 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/no-overlapping/emsd/emsd.fold1.csv'
    # fold2 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/no-overlapping/emsd/emsd.fold2.csv'
    # fold3 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/no-overlapping/emsd/emsd.fold3.csv'
    # :
    # fold1 = '/mnt/data/datasets/visem-dataset/03-02-02-02-features-in-folds-of-challenge-trackpy-displacement/emsd/emsd.fold1.csv'
    # fold2 = '/mnt/data/datasets/visem-dataset/03-02-02-02-features-in-folds-of-challenge-trackpy-displacement/emsd/emsd.fold2.csv'
    # fold3 = '/mnt/data/datasets/visem-dataset/03-02-02-02-features-in-folds-of-challenge-trackpy-displacement/emsd/emsd.fold3.csv'
    # 5s hop:
    # fold1 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/5s-hop/feature-vector-emsd-5s-hop-fold1.csv'
    # fold2 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/5s-hop/feature-vector-emsd-5s-hop-fold2.csv'
    # fold3 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/5s-hop/feature-vector-emsd-5s-hop-fold3.csv'
    
    # 1s hop:

    # local:
    # fold1 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/1s-hop/feature-vector-emsd-1s-hop-fold1.csv'
    # fold2 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/1s-hop/feature-vector-emsd-1s-hop-fold2.csv'
    # fold3 = '/mnt/data/datasets/visem-dataset/04-02-02-trackpy-feature-vectors-in-folds-of-challenge-plus-patient-data-results/1s-hop/feature-vector-emsd-1s-hop-fold3.csv'
    
    
    #optimize_parameters(args.feature_files, fold_labels, remove_labels=remove_labels, n_iter=args.random_iterations, use_prd=True, model_dir=args.model_dir)
    
    #print(f'All MAEs: {all_maes} mean: {sum(all_maes)/len(all_maes)}, all RMSEs: {all_rmses} mean: {sum(all_rmses)/len(all_rmses)}')
    def load_features(path):
        df = pd.read_csv(path)
        return df.values

    def tf_load_data(path):
        data = tf.io.read_file(path)
        n_columns = tf.strings.split(tf.strings.split(data, '\n')[0], ',').shape
       
        return n_columns
    # nas:
    fold1 = './features/imsd/5s-hop/imsd/fold_1'
    fold2 = './features/imsd/5s-hop/imsd/fold_2'
    fold3 = './features/imsd/5s-hop/imsd/fold_3'
    semen_analysis_data = './labels/semen_analysis_data.csv'
    semen_data = pd.read_csv(semen_analysis_data, delimiter=';', decimal=',')

    
    def get_patient_id(f):
        return splitext(splitext(basename(f))[0])[0].split('_')[0]


    def generator(fold):
        all_files = list(glob.glob(join(fold.decode('utf-8'), '*.csv')))
        for f in all_files:
            df = pd.read_csv(f, index_col=0).fillna(0)
            patient_id = get_patient_id(f)
            patient_semen_data = semen_data.loc[semen_data['ID'] == int(patient_id)]
            prog = patient_semen_data['Progressive motility (%)'].values[0]
            nonprog = patient_semen_data['Non progressive sperm motility (%)'].values[0]
            immotile = patient_semen_data['Immotile sperm (%)'].values[0]
            yield df.values, np.array([prog, nonprog, immotile], dtype=np.float32)
    
    fold1 = tf.data.Dataset.from_generator(generator, args=[fold1], output_types=(tf.float32, tf.float32)).cache().shuffle(100).padded_batch(64, padded_shapes=([500, None], [None])).prefetch(tf.data.experimental.AUTOTUNE)
    fold2 = tf.data.Dataset.from_generator(generator, args=[fold2], output_types=(tf.float32, tf.float32)).cache().shuffle(100).padded_batch(64, padded_shapes=([500, None], [None])).prefetch(tf.data.experimental.AUTOTUNE)
    fold3 = tf.data.Dataset.from_generator(generator, args=[fold3], output_types=(tf.float32, tf.float32)).cache().shuffle(100).padded_batch(64, padded_shapes=([500, None], [None])).prefetch(tf.data.experimental.AUTOTUNE)
    
    train_dataset = fold1.concatenate(fold2)
    test_dataset = fold3

    def build_mlp():
        model = keras.Sequential([tf.keras.layers.Permute((2,1), input_shape=(500, None))])
        model.add(layers.Masking(mask_value=0))
        for i in range(2):
            model.add(layers.TimeDistributed(layers.Dense(256)))
            model.add(layers.TimeDistributed(layers.Activation('relu')))
            model.add(layers.TimeDistributed(layers.BatchNormalization()))
            model.add(layers.TimeDistributed(layers.Dropout(0.2)))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(layers.Dense(3))
        optimizer = tf.keras.optimizers.SGD(1e-3)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model

    def build_cnn(hparams=None):
        model = keras.Sequential([
        tf.keras.layers.Permute((2,1), input_shape=(500, None)),
        layers.Reshape(target_shape=(-1, 500, 1)),
        layers.Convolution2D(32, (1, 3), strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.Dropout(0.5),
        layers.Convolution2D(64, (1, 3), strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((1,2)),
        layers.Dropout(0.5),
        layers.Convolution2D(128, (1, 3), strides=1, activation=None),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(3, #activation='sigmoid'
        )
        ])
        optimizer = tf.keras.optimizers.Nadam(1e-3)
        model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
        return model
    
    model_dir = 'test-model'
    model = build_mlp()
    model.summary()
    
    EPOCHS = 500
    
    #early_stop = keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=patience, restore_best_weights=True, mode='min')

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=join(model_dir, 'model.h5'), monitor='val_mean_absolute_error', save_best_only=False, period=20, mode='min')

    tb_callback = keras.callbacks.TensorBoard(log_dir=join(model_dir, 'logs'))
    history = model.fit(train_dataset, shuffle=True,
                        epochs=EPOCHS, validation_data=test_dataset, verbose=1, 
                        callbacks=[#tfdocs.modeling.EpochDots(), 
                        model_checkpoint, tb_callback])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    loss, mae, mse = model.evaluate(test_dataset, verbose=2)

    print("Test set Mean Abs Error: {:5.4f}".format(mae))

    # devel_predictions = model.predict(devel_dataset)

    rmse = math.sqrt(mse)