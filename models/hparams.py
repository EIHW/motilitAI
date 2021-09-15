import tensorflow as tf
from os.path import join
from tensorboard.plugins.hparams import api as hp

HP_BOW_SIZE = hp.HParam('bow_size', hp.Discrete([2500, 5000, 10000, 20000]))
HP_BOW_ASSIGNED_VECTORS = hp.HParam('bow_assigned_vectors',
                                    hp.Discrete([1, 10, 50, 100, 200, 500]))
HP_BOW = [HP_BOW_SIZE, HP_BOW_ASSIGNED_VECTORS]

PARAM_GRID_BOW = {hp.name: hp.domain.values for hp in HP_BOW}

# For MLP CNN and LSTM
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([256, 512, 1024]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 4]))
#HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 4, 8]))
#HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2]))

HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([3]))
HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([32, 64]))
HP_NUM_EPOCHS = hp.HParam('num_epochs', hp.Discrete([10000]))

HP_RECURRENT_DROPOUT = hp.HParam('recurrent_dropout', hp.Discrete([0.0, 0.2, 0.4]))
HP_CELL_TYPE = hp.HParam('cell_type', hp.Discrete(['gru', 'lstm']))
HP_BIDIRECTIONAL = hp.HParam('bidirectional', hp.Discrete([True, False]))
HP_NUM_RECURRENT_UNITS = hp.HParam('num_recurrent_units', hp.Discrete([64, 128, 256]))

HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64]))
HP_REGULARIZER = hp.HParam('kernel_regularizer',
                           hp.Discrete([1e-2, 1e-3, 1e-4]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu', 'elu']))
HP_DILATION_RATE = hp.HParam('dilation_rate', hp.Discrete([1]))

HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2, 0.4]))
#HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0]))

#HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-2]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4]))
#HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-4]))

# For SVR
HP_COST = hp.HParam(
    'C',
    hp.Discrete([
        1000.0, 100.0, 10.0, 1.0, 0.1
        ]))

HP_SVR = [HP_COST]
PARAM_GRID_SVR = {hp.name: hp.domain.values for hp in HP_SVR}


METRIC_MEAN_MAE = 'mean_mae'
METRIC_MEAN_RMSE = 'mean_rmse'
MAE_FOLDS = ['mean_mae_fold_1', 'mean_mae_fold_2', 'mean_mae_fold_3']
RMSE_FOLDS = ['mean_rmse_fold_1', 'mean_rmse_fold_2', 'mean_rmse_fold_3']

METRIC_MEAN_MAE_VAL = 'mean_mae_val'
METRIC_MEAN_RMSE_VAL = 'mean_rmse_val'
MAE_FOLDS_VAL = [
    'mean_mae_fold_1_val', 'mean_mae_fold_2_val', 'mean_mae_fold_3_val'
]
RMSE_FOLDS_VAL = [
    'mean_rmse_fold_1_val', 'mean_rmse_fold_2_val', 'mean_rmse_fold_3_val'
]

METRIC_BEST_EPOCH = [
    'best_epoch_fold_1', 'best_epoch_fold_2', 'best_epoch_fold_3'
]

HP_ALL = [
    HP_NUM_EPOCHS, HP_LEARNING_RATE, HP_ACTIVATION, HP_NUM_LAYERS,
    HP_BATCHSIZE, HP_DROPOUT, HP_REGULARIZER
]
HP_MLP = HP_ALL + [HP_NUM_UNITS]
HP_CNN = HP_ALL + [HP_KERNEL_SIZE, HP_NUM_FILTERS, HP_DILATION_RATE]
HP_RNN = HP_ALL + [HP_NUM_RECURRENT_UNITS, HP_CELL_TYPE, HP_RECURRENT_DROPOUT, HP_BIDIRECTIONAL]

PARAM_GRID_MLP = {hp.name: hp.domain.values for hp in HP_MLP}
PARAM_GRID_CNN = {hp.name: hp.domain.values for hp in HP_CNN}
PARAM_GRID_RNN = {hp.name: hp.domain.values for hp in HP_RNN}



def setup_hparams(model_dir='logs', model_type='mlp', bow=False):
    if model_type == 'mlp':
        all_hparams = HP_MLP
        param_grid = PARAM_GRID_MLP
    elif model_type == 'cnn':
        all_hparams = HP_CNN
        param_grid = PARAM_GRID_CNN
    elif model_type == 'rnn':
        all_hparams = HP_RNN
        param_grid = PARAM_GRID_RNN
    elif model_type == 'svr':
        all_hparams = HP_SVR
        param_grid = PARAM_GRID_SVR
    if bow:
        all_hparams += HP_BOW
    with tf.summary.create_file_writer(join(model_dir, 'logs',
                                            'hparam_tuning')).as_default():
        hp.hparams_config(
            hparams=all_hparams,
            metrics=[
                hp.Metric(METRIC_MEAN_MAE,
                          display_name='Mean MAE (across folds)'),
                hp.Metric(METRIC_MEAN_RMSE,
                          display_name='Mean RMSE (across folds)'), *[
                              hp.Metric(m, display_name=f'MAE fold {i+1}')
                              for i, m in enumerate(MAE_FOLDS)
                          ], *[
                              hp.Metric(m, display_name=f'RMSE fold {i+1}')
                              for i, m in enumerate(RMSE_FOLDS)
                          ],
                hp.Metric(METRIC_MEAN_MAE_VAL,
                          display_name='Mean validation MAE (across folds)'),
                hp.Metric(METRIC_MEAN_RMSE_VAL,
                          display_name='Mean validation RMSE (across folds)'),
                *[
                    hp.Metric(m, display_name=f'validation MAE fold {i+1}')
                    for i, m in enumerate(MAE_FOLDS_VAL)
                ], *[
                    hp.Metric(m, display_name=f'validation RMSE fold {i+1}')
                    for i, m in enumerate(RMSE_FOLDS_VAL)
                ], *[
                    hp.Metric(m, display_name=f'Best Epoch fold {i+1}')
                    for i, m in enumerate(METRIC_BEST_EPOCH)
                ]
            ])
    return param_grid


def write_metrics(all_mae, all_rmse, all_val_mae, all_val_rmse, all_best_epoch,
                  hparams, model_dir):
    with tf.summary.create_file_writer(model_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mean_mae = sum(all_mae) / len(all_mae)
        mean_rmse = sum(all_rmse) / len(all_rmse)
        mean_mae_val = sum(all_val_mae) / len(all_val_mae)
        mean_rmse_val = sum(all_val_rmse) / len(all_val_rmse)

        tf.summary.scalar(METRIC_MEAN_MAE, mean_mae, step=1)
        tf.summary.scalar(METRIC_MEAN_RMSE, mean_rmse, step=1)
        tf.summary.scalar(METRIC_MEAN_MAE_VAL, mean_mae_val, step=1)
        tf.summary.scalar(METRIC_MEAN_RMSE_VAL, mean_rmse_val, step=1)
        for i in range(len(all_mae)):
            tf.summary.scalar(MAE_FOLDS[i], all_mae[i], step=1)
            tf.summary.scalar(RMSE_FOLDS[i], all_rmse[i], step=1)
            tf.summary.scalar(MAE_FOLDS_VAL[i], all_val_mae[i], step=1)
            tf.summary.scalar(RMSE_FOLDS_VAL[i], all_val_rmse[i], step=1)
            tf.summary.scalar(METRIC_BEST_EPOCH[i], all_best_epoch[i], step=1)
        print(
            f'validation: All MAEs: {all_val_mae} mean: {mean_mae_val}, all RMSEs: {all_val_rmse} mean: {mean_rmse_val}'
        )
        print(
            f'test: All MAEs: {all_mae} mean: {mean_mae}, all RMSEs: {all_rmse} mean: {mean_rmse}'
        )