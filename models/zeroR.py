import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEMEN_DATA = '/mnt/data/datasets/visem-dataset/01-labels-stats/semen_analysis_data.csv'

PATIENT_RELATED = ['abstinence', 'age']
MOTILITY_LABELS = [
    'Progressive motility (%)', 'Non progressive sperm motility (%)',
    'Immotile sperm (%)'
]
MORPHOLOGY_LABELS = [
    'Head defects (%)', 'Midpiece and neck defects (%)', 'Tail defects (%)'
]


def zero_r(fold_files, fold_labels):
    semen_data = pd.read_csv(SEMEN_DATA, delimiter=';',
                             decimal=',').set_index('ID')

    train_dataset = pd.concat([
        pd.read_csv(fold_files[0], delimiter=';', header=None),
        pd.read_csv(fold_files[1], delimiter=';', header=None)
    ],
                              ignore_index=True)

    test_dataset = pd.read_csv(fold_files[2], delimiter=';', header=None)

    ids_train = list(
        train_dataset.pop(
            train_dataset.columns[0]).apply(lambda x: x[1:-1]).astype(int))
    patient_semen_data_train = semen_data.loc[ids_train]
    ids_test = list(
        test_dataset.pop(
            test_dataset.columns[0]).apply(lambda x: x[1:-1]).astype(int))
    patient_semen_data_test = semen_data.loc[ids_test]

    train_labels = np.stack([
        patient_semen_data_train.pop(fold_label) for fold_label in fold_labels
    ],
                            axis=-1)

    test_labels = np.stack([
        patient_semen_data_test.pop(fold_label) for fold_label in fold_labels
    ],
                           axis=-1)

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    train_dataset = norm(train_dataset)
    test_dataset = norm(test_dataset)

    means = [train_labels.mean(axis=0)]
    #print(train_labels)
    #print(means)
    preds = np.repeat(means, [len(test_labels),], axis=0)
    #print(preds)
    stack = np.concatenate([np.expand_dims(ids_test, -1), preds, test_labels],
                           axis=-1)
    df = pd.DataFrame(
        stack,
        columns=['id', 'pred1', 'pred2', 'pred3', 'true1', 'true2', 'true3'])
    df[['pred1', 'pred2', 'pred3', 'true1', 'true2',
        'true3']] = df[['pred1', 'pred2', 'pred3', 'true1', 'true2',
                        'true3']].apply(pd.to_numeric)
    df = df.groupby('id').mean()
    preds = df[['pred1', 'pred2', 'pred3']].values
    test_labels = df[['true1', 'true2', 'true3']].values
    mae = mean_absolute_error(test_labels, preds)
    mse = mean_squared_error(test_labels, preds)
    print("Test set Mean Abs Error: {:5.4f}".format(mae))
    
    # devel_predictions = model.predict(devel_dataset)

    rmse = math.sqrt(mse)
    print("Test set RMSE Error: {:5.4f}".format(rmse))

    return mae, rmse


if __name__ == '__main__':
    fold_files = [
        '/mnt/data/datasets/visem-dataset/05-bow-features/imsd/5s-hop/BoW/2500/1/imsd_feature_vectors_fold_1.csv',
        '/mnt/data/datasets/visem-dataset/05-bow-features/imsd/5s-hop/BoW/2500/1/imsd_feature_vectors_fold_2.csv',
        '/mnt/data/datasets/visem-dataset/05-bow-features/imsd/5s-hop/BoW/2500/1/imsd_feature_vectors_fold_3.csv'
    ]
    mae1, rmse1 = zero_r(fold_files=[fold_files[1], fold_files[2], fold_files[0]], fold_labels=MORPHOLOGY_LABELS)
    mae2, rmse2 = zero_r(fold_files=[fold_files[0], fold_files[2], fold_files[2]], fold_labels=MORPHOLOGY_LABELS)
    mae3, rmse3 = zero_r(fold_files=fold_files, fold_labels=MORPHOLOGY_LABELS)
    
    print(f'Mean MAE: {(mae1 + mae2 + mae3) / 3}')
    print(f'Mean RMSE: {(rmse1 + rmse2 + rmse3) / 3}')