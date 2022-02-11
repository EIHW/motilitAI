#!/usr/bin/env python
from __future__ import division, unicode_literals, print_function

import matplotlib as mpl
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import cv2 as cv
import pims
import trackpy as tp
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from os.path import join, basename, splitext, exists, relpath, dirname
from glob import glob
from tqdm import tqdm
import sys
import yaml
from os import makedirs

mpl.rc('figure', figsize=(10, 5))
mpl.rc('image', cmap='gray')


def extract_displacement_features(
        input_video,
        output_base,
        fold,
        hop_size=1,
        window=10,
        pad="constant"):
    id = basename(input_video).split("_")[0]
    scene = splitext(basename(input_video))[0].split("-")[-1]
    makedirs(output_base, exist_ok=True)
    makedirs(join(output_base, 'imsd'), exist_ok=True)
    makedirs(join(output_base, 'emsd'), exist_ok=True)

    cam = cv.VideoCapture(input_video)
    frames = []
    while True:
        _ret, frame = cam.read()
        if not _ret:
            break
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames.append(frame_gray)
    tp.quiet()
    k = True
    i = 0
    window *= 50
    while k:
        if i <= (len(frames) - window):

            f = tp.batch(frames[i:(i + window)],
                         11,
                         minmass=900,
                         invert=False,
                         processes='auto',
                         engine='numba')

            dataframe = tp.link(f, 5, memory=3)
            dataframe_filtered = tp.filter_stubs(dataframe, 25)

            drift = tp.compute_drift(dataframe_filtered)
            dataframe_without_drift = tp.subtract_drift(
                dataframe_filtered.copy(), drift)

            # individual mean squared displacement (MSD):
            if len(dataframe_without_drift) < 1:
                print('No detected track in window. Skipping.')
                i += 50 * hop_size
                continue
            dataframe_imsd = tp.imsd(
                dataframe_without_drift, 100 / 285., 50, max_lagtime=window
            )  # microns per pixel = 100/285., frames per second = 50

            # ensemble mean squared displacement (EMSD):
            dataframe_emsd = tp.emsd(
                dataframe_without_drift, 100 / 285., 50, max_lagtime=window
            )  # microns per pixel = 100/285., frames per second = 50

            # create dataframe csv files
            #video_basename = basename(splitext(input_video)[0])
            transposed_emsd = transpose_feature_vector(dataframe_emsd.to_frame(), pad=pad, window=window)
            _columns = transposed_emsd.columns
            transposed_emsd["id_scene"] = f"{id}_{scene}"
            transposed_emsd = transposed_emsd[["id_scene"] + list(_columns)]
            transposed_imsd = transpose_feature_vector(dataframe_imsd, pad=pad, window=window)
            _columns = transposed_imsd.columns
            transposed_imsd["ID"] = id
            transposed_imsd = transposed_imsd[["ID"] + list(_columns)]
            imsd_path = join(output_base, "imsd", f"{fold}.csv")
            concat_imsd_path = join(output_base, "imsd", "features.csv")
            emsd_path = join(output_base, "emsd", f"{fold}.csv")
            transposed_imsd.to_csv(
                imsd_path, index=False, sep=";", mode='a', header=not exists(imsd_path)
            )
            transposed_imsd.to_csv(
                concat_imsd_path, index=False, sep=";", mode='a', header=not exists(imsd_path)
            )
            transposed_emsd.to_csv(
                emsd_path, index=False, sep=",", mode='a', header=not exists(emsd_path)
            )

            i += 50 * hop_size
        else:
            k = False



def transpose_feature_vector(df, pad="constant", window=500):
    df.index = [f"{i:.2f}" for i in df.index]
    index = [f"{i:.2f}" for i in range(1, window+1)]
    df = df.reindex(index=index)
    if pad == "constant":
        df = df.fillna(-1)
    elif pad == "same":
        df = df.ffill(axis=0)
    df_transposed = df.transpose()

    df_transposed.columns = list(map(lambda x: f'lag_time_{x}', df.index))
    return df_transposed



if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["features"]["displacement"]
    basepath = "./data/processed/cut-videos"
    output_base = "./features/displacement"

    videos = sorted(list(glob(f'{basepath}/**/*.mp4')))
    for v in tqdm(videos):
        fold = f"{dirname(relpath(v, basepath))}"
        extract_displacement_features(input_video=v,
                                      hop_size=params["hop"],
                                      output_base=output_base,
                                      fold=fold,
                                      window=params["window"],
                                      pad=params["padding"])

