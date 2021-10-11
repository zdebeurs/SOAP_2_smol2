import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl
from Weighted_LS_fit import Weighted_LS_fit

def Weighted_LS_crossval_fit(ccf_indexes, ccf_list_np, rv_np, s_rv_np):
    # Divide data into training and validation sets
    # print("og list: "+str(s_rv_np))
    y_val_preds_list = []
    a_coeff_list = []
    new_train_rms_list = []

    for i in range(0, len(rv_np)):
        train_rv_np = np.delete(rv_np, i)
        train_s_rv_np = np.delete(s_rv_np, i)
        train_ccf_list_np = np.delete(ccf_list_np, i, 0)
        train_ccf_list_np_transpose = train_ccf_list_np.transpose()
        #train_time_np = np.delete(time_np, i)

        val_rv_np = rv_np[i]
        val_s_rv_np = s_rv_np[i]
        val_ccf_list_np = ccf_list_np[i]
        val_ccf_list_np_transpose = np.array([val_ccf_list_np]).transpose()
        #train_time_np = time_np[i]

        # double checks if the validation example is excluded from the training set
        #if val_rv_np in train_rv_np:
        #    print("val rv example is in training set!")
        #if val_s_rv_np in train_s_rv_np:
        #    print("val s_rv example is in training set!")
        if any(x in val_ccf_list_np.tolist() for x in [list(ccf_list_np[i])]):
            print("val ccf example is in training set!")

        a_coeff, y_train_preds, CCF_train_matrix, raw_train_rms, new_train_rms = Weighted_LS_fit(ccf_indexes,
                                                                                                 train_ccf_list_np,
                                                                                                 train_ccf_list_np_transpose,
                                                                                                 train_rv_np,
                                                                                                 train_s_rv_np)
        a_coeff_list.append(a_coeff)
        new_train_rms_list.append(new_train_rms)
        n_rows = len([val_ccf_list_np])
        CCF_val_matrix = np.zeros((n_rows, len(ccf_indexes)))
        for j in np.arange(0, len(ccf_indexes)):
            CCF_val_matrix[:, j] = val_ccf_list_np_transpose[ccf_indexes[j]]  # (for all observations)

        y_val_preds = (CCF_val_matrix.dot(a_coeff[1:]) + a_coeff[0]).tolist()[0]
        y_val_preds_list.append(y_val_preds)

    y = rv_np
    # Compute the scatter metric
    raw_val_rms = np.std(y, ddof=1)
    new_val_rms = np.std(y - y_val_preds_list, ddof=1)

    return y_val_preds_list, raw_val_rms, new_val_rms, new_train_rms_list, a_coeff_list

if __name__ == '__main__':
    Weighted_LS_crossval_fit()