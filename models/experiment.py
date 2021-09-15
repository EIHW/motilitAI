import pandas as pd
import numpy as np
import math
import tensorflow as tf
from os.path import join
from learner import MLPLearner, CNNLearner, SVRLearner, RNNLearner
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hparams import *
from os import makedirs

SEMEN_DATA = 'semen_analysis_data.csv'

MOTILITY_LABELS = [
    'Progressive motility (%)', 'Non progressive sperm motility (%)',
    'Immotile sperm (%)'
]
MORPHOLOGY_LABELS = [
    'Head defects (%)', 'Midpiece and neck defects (%)', 'Tail defects (%)'
]

def get_sequence_data(df):
    label_columns = [c for c in df.columns if str(c).startswith('label')]
    g = df.groupby(by=['ID', 'scene'])
    groups = [g.get_group(k) for k in g.groups.keys()]
    labels = np.array([np.stack((g.pop(f'label_{i}').values[0] for i in range(len(label_columns))), axis=-1) for g in groups])
    ids = np.array([g.pop('ID').values[0] for g in groups])

    data = []
    for g in groups:
        g.pop('scene')
        data.append(g.values) 
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(data, value=-1.0, dtype='float32')
    return padded_data, labels, ids


def run_fold(fold_dataframes, hparams, model_type='mlp', fold_labels='motility', remove_labels=[], use_prd=False,  model_dir='model', semen_data=SEMEN_DATA, time_series=False):
    semen_data = pd.read_csv(SEMEN_DATA, delimiter=';',
                             decimal=',').set_index('ID')
    folds_copy = [df.copy() for df in fold_dataframes]
    train_dataset = pd.concat([folds_copy[0], folds_copy[1]], ignore_index=True)
    if not time_series:
        train_dataset = train_dataset.sort_values([train_dataset.columns[0]])
    test_dataset = folds_copy[2]

    ids_train = train_dataset.pop(train_dataset.columns[0])
    ids_test = test_dataset.pop(test_dataset.columns[0])

    if time_series:
        scenes_train = train_dataset.pop('scene')
        scenes_test = test_dataset.pop('scene')

    patient_semen_data_train = semen_data.loc[ids_train]
    patient_semen_data_test = semen_data.loc[ids_test]

    if fold_labels == 'motility':
        fold_labels = MOTILITY_LABELS
    elif fold_labels == 'morphology':
        fold_labels = MORPHOLOGY_LABELS

    train_labels = np.stack([
        patient_semen_data_train.pop(fold_label) for fold_label in fold_labels
    ],
        axis=-1)
    
    test_labels = np.stack([
        patient_semen_data_test.pop(fold_label) for fold_label in fold_labels
    ],
        axis=-1)


    
    for remove_label in remove_labels:
        train_dataset.pop(remove_label)
        test_dataset.pop(remove_label)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    
    def norm(x):
        train_stats[train_stats['std'] == 0] = 1
        return (x - train_stats['mean']) / train_stats['std']

    train_dataset = norm(train_dataset)
    test_dataset = norm(test_dataset)
    train_dataset = train_dataset.values
    test_dataset = test_dataset.values

    if time_series:
        train_df = pd.DataFrame(data=train_dataset)
        train_df['ID'] = ids_train
        train_df['scene'] = scenes_train

        test_df = pd.DataFrame(data=test_dataset)
        test_df['ID'] = ids_test
        test_df['scene'] = scenes_test
        for i in range(train_labels.shape[1]):
            train_df[f'label_{i}'] = train_labels[:, i]
            test_df[f'label_{i}'] = test_labels[:, i]
        
        train_dataset, train_labels, ids_train = get_sequence_data(train_df)
        test_dataset, test_labels, ids_test = get_sequence_data(test_df)

   
    if model_type == 'mlp':
        model = MLPLearner(model_dir=model_dir, hparams=hparams)

    elif model_type == 'cnn':
        model = CNNLearner(model_dir=model_dir, hparams=hparams)

    elif model_type == 'rnn':
        model = RNNLearner(model_dir=model_dir, hparams=hparams)
    
    elif model_type == 'svr':
        model = SVRLearner(model_dir=model_dir, hparams=hparams, cv=5)
    model.fit(train_dataset, train_labels)
    preds = model.predict(test_dataset)
    stack = np.concatenate([np.expand_dims(ids_test, -1), preds, test_labels], axis=-1)
    df = pd.DataFrame(stack, columns=['id', 'pred1', 'pred2', 'pred3', 'true1', 'true2', 'true3'])
    df[['pred1', 'pred2', 'pred3', 'true1', 'true2', 'true3']] = df[['pred1', 'pred2', 'pred3', 'true1', 'true2', 'true3']].apply(pd.to_numeric)
    df = df.groupby('id').mean()
    preds = df[['pred1', 'pred2', 'pred3']].values
    test_labels = df[['true1', 'true2', 'true3']].values
    mae = mean_absolute_error(test_labels, preds)
    mse = mean_squared_error(test_labels, preds)
    print("Test set Mean Abs Error: {:5.4f}".format(mae))
    makedirs(model_dir, exist_ok=True)
    df.to_csv(join(model_dir, 'predictions.csv'), index=False)
    rmse = math.sqrt(mse)
    model.reset()
    return mae, rmse, model.val_mae, model.val_rmse, model.best_epoch


def run_cross_validation(fold_files, hparams, fold_labels='motility', remove_labels=[], model_type='mlp', use_prd=True, model_dir='model', semen_data=SEMEN_DATA, time_series=False):
    all_mae = []
    all_rmse = []
    all_val_mae = []
    all_val_rmse = []
    all_best_epoch = []

    # fold_maes = []
    # fold_rmses = []
    for i in range(len(fold_files)):
        print(
            f'-------------------------------------------- fold {i+1} -----------------------------------'
        )
        folds = list(range(len(fold_files)))
        test_file = fold_files[i]
        folds.pop(i)
        # fold_maes = []
        # fold_rmses = []
        # for j in range(len(folds)):
        #     remaining_folds = folds.copy()
        train_file = fold_files[folds[0]]
        devel_file = fold_files[folds[1]]
        # print(train_file, devel_file, test_file)
        mae, rmse, val_mae, val_rmse, best_epoch = run_fold([train_file, devel_file, test_file],
            fold_labels=fold_labels,
            hparams=hparams,
            remove_labels=remove_labels,
            model_type=model_type,
            use_prd=use_prd,
            model_dir=join(model_dir, f'fold-{i+1}'),
            semen_data=semen_data,
            time_series=time_series)
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_val_mae.append(val_mae)
        all_val_rmse.append(val_rmse)
        all_best_epoch.append(best_epoch)

    write_metrics(all_mae, all_rmse, all_val_mae, all_val_rmse, all_best_epoch, hparams, model_dir)