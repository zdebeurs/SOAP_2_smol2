import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl

from astropy.io import fits
import pandas as pd
from sklearn.model_selection import KFold  # import KFold


def k_fold_crossval_program(ccf_indexes, ccf_list_np, ccf_list_np_transpose, rv_np, num_folds, bjd_list)

    n_cols = len(ccf_indexes) + 1
    n_rows = len(ccf_list_np)  # len(ccf_list_np)<---- unsmoothed
    x = np.zeros((n_rows, n_cols))

    x[:, 0] = 1
    # x[:,1] =  np.sin(2*np.pi*(time_np-T_c)/period+np.pi) #try a whole bunch periods
    # x[:,4] =  np.cos(2*np.pi*(time_np-T_c)/period)
    for i in np.arange(0, len(ccf_indexes)):
        x[:, i + 1] = ccf_list_np_transpose[ccf_indexes[i]]

    X = np.array(x.tolist())  # create an array
    y = np.array(rv_np)  # Create another array
    kf = KFold(n_splits=num_folds, shuffle=True)  # Define the split - into 2 folds
    kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator
    # print(kf)

    y_val_preds_list = []
    bjd_val_list = []

    for train_index, test_index in kf.split(X):
        # print('TRAIN:'+str(train_index)+'TEST:'+ str(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        bjd_train, bjd_test = bjd_list[train_index], bjd_list[test_index]
        y_err_train, y_err_test = s_rv_np[train_index], s_rv_np[test_index]

        a_coeff, y_train_preds, CCF_train_matrix, raw_train_rms, new_train_rms = Weighted_simple_crossval_LS_fit(
            ccf_indexes,
            X_train,
            y_train,
            y_err_train)
        y_val_preds = (X_test.dot(a_coeff)).tolist()
        y_val_preds_list = np.concatenate((y_val_preds_list, y_val_preds))
        bjd_val_list = np.concatenate((bjd_val_list, bjd_test))

    # create pandas dataframe
    df_val = pd.DataFrame(list(zip(bjd_val_list,
                                   y_val_preds_list)),
                          columns=["BJD",
                                   "vrad_preds"])
    df_val_sorted = df_val.sort_values(by=['BJD'])

    # Compute the scatter metric
    raw_rms =  np.std(y, ddof=1)
    ccf_corr_rms = np.std(y - df_val["vrad_preds"], ddof=1)

    return df_val["vrad_preds"], raw_rms, ccf_corr_rms


if __name__ == '__main__':
    k_fold_crossval_program()