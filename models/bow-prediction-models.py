from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import argparse
from os.path import join, basename
from sklearn.model_selection import ParameterSampler, ParameterGrid
from tqdm import tqdm
import tensorflow
import gc
from hparams import *
from experiment import run_cross_validation
from tensorflow.compat.v1.keras.backend import set_session, clear_session, get_session


def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model  # this is from global space - change this as you need
    except:
        pass

    # if it's done something you should see a number being outputted
    print(gc.collect())

    # use the same config as you used to create the session
    config = tensorflow.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.compat.v1.Session(config=config))


PATIENT_RELATED = ['abstinence', 'age']
MOTILITY_LABELS = [
    'Progressive motility (%)', 'Non progressive sperm motility (%)',
    'Immotile sperm (%)'
]
MORPHOLOGY_LABELS = [
    'Head defects (%)', 'Midpiece and neck defects (%)', 'Tail defects (%)'
]

def optimize_parameters(bow_dirname,
                        fold_files,
                        fold_labels,
                        n_iter=10,
                        remove_labels=MORPHOLOGY_LABELS,
                        use_prd=True,
                        model_dir='model',
                        model_type='mlp',
                        initial_session=0):
    param_grid = setup_hparams(model_dir=model_dir, model_type=model_type, bow=True)
    session_num = initial_session
    param_list = list(
        ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    bow_param_list = list(ParameterGrid(PARAM_GRID_BOW))
    for bow_params in tqdm(bow_param_list):
        for hparams in tqdm(param_list):
            hparams.update(bow_params)
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h: hparams[h] for h in hparams})
            fold_files_bow = [
                join(bow_dirname, str(hparams[HP_BOW_SIZE.name]),
                    str(hparams[HP_BOW_ASSIGNED_VECTORS.name]), fold_file)
                for fold_file in fold_files
            ]
            print(fold_files_bow)
            fold_dataframes = [pd.read_csv(f, header=None, delimiter=';') for f in fold_files_bow]
            for df in fold_dataframes:
                df[df.columns[0]] = df[df.columns[0]].apply(lambda x: x[1:-1]).astype(int)

            run_cross_validation(fold_dataframes,
                                        fold_labels=fold_labels,
                                        hparams=hparams,
                                        use_prd=use_prd,
                                        model_type=model_type,
                                        model_dir=join(model_dir, 'logs',
                                                       'hparam_tuning',
                                                       run_name))
            session_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate model on 3-fold emsd cross-validation.'
    )
    parser.add_argument('feature_files',
                        metavar='X',
                        type=str,
                        nargs=3,
                        help='Supply names of feature files for fold 1, 2 and 3.')
    parser.add_argument('-bd', '--bow-dirname',
                        type=str,
                        required=True,
                        help='Supply BoW directory name.')
    parser.add_argument(
        '-ma',
        '--model-architecture',
        type=str,
        choices=['mlp', 'cnn', 'svr'],
        default='mlp',
        help='Type of neural network architecture (default: CNN)')
    parser.add_argument('-md',
                        '--model-dir',
                        type=str,
                        default='model',
                        help='Model directory.')
    parser.add_argument('-s',
                        '--sizes',
                        type=int,
                        default=[2500, 5000],
                        help='Sizes for BoW.',
                        nargs='+')
    parser.add_argument('-a',
                        '--assignment-vectors',
                        type=int,
                        default=[50],
                        help='Number of assigned codebook vectors.',
                        nargs='+')
    parser.add_argument('-sn',
                        '--session-number',
                        type=int,
                        default=0,
                        help='Starting session.')
    parser.add_argument('-ri',
                        '--random-iterations',
                        type=int,
                        default=10,
                        help='Number of random grid serach iterations.')
    parser.add_argument('-l',
                        '--target-labels',
                        type=str,
                        choices=['motility', 'morphology'],
                        default='motility',
                        help='Target labels (default: motility)')
    args = parser.parse_args()

    assert all(values in PARAM_GRID_BOW[HP_BOW_SIZE.name] for values in args.sizes) and all(values in PARAM_GRID_BOW[HP_BOW_ASSIGNED_VECTORS.name]
                                                                                            for values in args.assignment_vectors), f'Given codebook sizes and assigned vectors do not match hparams domain values!'
    PARAM_GRID_BOW[HP_BOW_SIZE.name] = args.sizes
    PARAM_GRID_BOW[HP_BOW_ASSIGNED_VECTORS.name] = args.assignment_vectors

    feature_files = sorted([basename(f) for f in args.feature_files])
    optimize_parameters(args.bow_dirname,
                        feature_files,
                        fold_labels=args.target_labels,
                        model_type=args.model_architecture,
                        n_iter=args.random_iterations,
                        use_prd=True,
                        model_dir=args.model_dir,
                        initial_session=args.session_number)
