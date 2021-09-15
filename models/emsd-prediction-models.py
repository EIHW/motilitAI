from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import argparse
from tensorboard.plugins.hparams import api as hp
from os.path import join
from sklearn.model_selection import ParameterSampler
from hparams import *
from experiment import run_cross_validation

PATIENT_RELATED = ['abstinence', 'age']
MOTILITY_LABELS = ['immotile_sperm', 'progressive_motility', 'non_progressive_motility']
MORPHOLOGY_LABELS = ['head_defects', 'mid_defects', 'tail_defects']


def optimize_parameters(fold_dataframes, fold_labels, n_iter=10, use_prd=True, semen_data='semen_data_analysis.csv', model_dir='model', model_type='mlp'):
    param_grid = setup_hparams(model_dir=model_dir, model_type=model_type, bow=False)
    session_num = 0
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=42))
    time_series = model_type == 'rnn'
    for hparams in param_list:
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        run_cross_validation(fold_dataframes, fold_labels=fold_labels, hparams=hparams, use_prd=use_prd, model_type=model_type, model_dir=join(model_dir, 'logs', 'hparam_tuning', run_name), semen_data=semen_data, time_series=time_series)
        session_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate model on 3-fold emsd cross-validation.')
    parser.add_argument('feature_files', metavar='X', type=str, nargs=3,
                        help='Supply emsd feature files for fold 1, 2 and 3.')
    parser.add_argument('-ma', '--model-architecture', type=str, choices=['mlp', 'cnn', 'svr', 'rnn'], default='mlp',
                        help='Type of neural network architecture (default: CNN)')
    parser.add_argument('-md', '--model-dir', type=str,  default='model',
    help='Model directory.')
    parser.add_argument('-ri', '--random-iterations', type=int,  default=10,
    help='Number of random grid serach iterations.')
    parser.add_argument('-l', '--target-labels', type=str, choices=['motility', 'morphology'], 
                        default='motility', help='Target labels (default: motility)')
    parser.add_argument('-sd', '--semen-data', type=str, default='sement_data_analysis.csv', help='Path to semen_data_analysis csv file.')
    args = parser.parse_args()
    fold_dataframes = [pd.read_csv(f) for f in args.feature_files]
    for df in fold_dataframes:
        ids = df[df.columns[0]].apply(lambda x: int(x.split('_')[0]))
        if args.model_architecture == 'rnn':
            scenes = df[df.columns[0]].apply(lambda x: int(x.split('_')[1]))
            df['scene'] = scenes

        df[df.columns[0]] = df[df.columns[0]].apply(lambda x: int(x.split('_')[0]))
        for l in PATIENT_RELATED + MOTILITY_LABELS + MORPHOLOGY_LABELS:
            if l in df.columns:
                df.pop(l) 

    optimize_parameters(fold_dataframes, fold_labels=args.target_labels, model_type=args.model_architecture, n_iter=args.random_iterations, use_prd=False, semen_data=args.semen_data, model_dir=args.model_dir)
    
  