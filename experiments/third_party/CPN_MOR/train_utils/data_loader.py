import numpy as np
from pathlib import Path


def myloader(config):
    path_data = config["data"]["train_data_path"]
    path_test_data = config["data"]["test_data_path"]
    path_test = Path(path_test_data)
    ntrain = config["data"]["ntrain"]
    ntest = config["data"]["ntest"]
    S_full = np.load(path_data)
    nx, ny, nz = S_full.shape[0], 1, 1
    if S_full.ndim == 3:
        ny = S_full.shape[1]
    if S_full.ndim == 4:
        nz = S_full.shape[2]

    S_full = S_full.reshape(nx * ny * nz, -1)

    S = S_full
    if ntrain != -1:
        S = S_full[:, :ntrain]

    if not path_test.exists():
        S_test = S_full[:, -ntest:]
    else:
        S_test = np.load(path_test_data)
        nx, ny, nz = S_full.shape[0], 1, 1
        if S_test.ndim == 3:
            ny = S_test.shape[1]
        if S_test.ndim == 4:
            nz = S_full.shape[2]
            S_test = S_full.reshape(nx * ny * nz, -1)
        if ntest != -1:
            S_test = S_test[:, :ntest]

    ntrain = S.shape[1]
    ntest = S_test.shape[1]

    sref = np.mean(S, axis=1)

    Sref = np.array([sref, ] * ntrain).T
    Sref_test = np.array([sref, ] * ntest).T

    print("Training data shape  =  ", S.shape)
    print("Test data shape   =  ", S_test.shape)

    return S, S_test, Sref, Sref_test
