#!/usr/bin/env python
import subprocess
import yaml
from os import makedirs
from os.path import join, basename, splitext
from tqdm import tqdm

PATH = './openXBOW'


def run_xbow_experiment(base_file,
                        labels=None,
                        files=[],
                        a_s=[1, 10 ,50, 100, 200, 500],
                        sizes=[2500, 5000, 10000, 20000],
                        output_path='./openXBoW-experiments',
                        **kwargs):
    for size in tqdm(sizes):
        for a in tqdm(a_s):
            if a < size:
                folder = join(output_path, str(size), str(a))
                makedirs(folder, exist_ok=True)
                create_bow(i=base_file,
                           o=join(folder,
                                  f'{splitext(basename(base_file))[0]}.csv'),
                           a=a,
                           B=join(folder, "codebook.txt"),
                           size=size,
                           l=labels,
                           **kwargs)
                for f in files:
                    apply_codebook(i=f,
                                   o=join(folder,
                                          f'{splitext(basename(f))[0]}.csv'),
                                   b=join(folder, "codebook.txt"),
                                   l=labels,
                                   norm=kwargs["norm"])


def create_bow(i,
               o,
               a,
               B,
               size,
               standardizeInput=False,
               standardizeOutput=False,
               normalizeInput=False,
               normalizeOutput=False,
               c="random",
               log=False,
               l=None,
               norm=None):
    command = [
        "java", "-Xms12g", "-jar", f"{PATH}/openXBOW.jar", "-i", i, "-o", o, "-a",
        str(a), "-c", c, "-B", B, "-size",
        str(size)
    ]
    if standardizeInput:
        command.append("-standardizeInput")
    elif normalizeInput:
        command.append("-normalizeInput")
    if standardizeOutput:
        command.append("-standardizeOutput")
    elif normalizeOutput:
        command.append("-normalizeOutput")
    if log:
        command.append("-log")
    if l:
        command.append("-l")
        command.append(l)
    if norm:
        command.append("-norm")
        command.append(str(norm))
    subprocess.run(command)
    return


def apply_codebook(i, o, b, l=None, norm=None):
    command = [
        "java", "-Xms12g", "-jar", f"{PATH}/openXBOW.jar", "-i", i, "-o", o, "-b", b,
        "-writeName"
    ]
    if l:
        command.append("-l")
        command.append(l)
    if norm:
        command.append("-norm")
        command.append(str(norm))
    subprocess.run(command)
    return


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["features"]["xbow"]
    run_xbow_experiment(
        base_file=
        './features/displacement/imsd/features.csv',
        output_path=
        './features/xbow/imsd',
        files=[
            './features/displacement/imsd/fold_1.csv',
            './features/displacement/imsd/fold_2.csv',
            './features/displacement/imsd/fold_3.csv'
        ],
        log=True,
        sizes=params["sizes"],
        a_s=params["assignment_vectors"],
        standardizeInput=params["standardize_input"],
        norm=None)
