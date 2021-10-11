import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl
import radvel


def Weighted_LS_fit_with_h_alpha_AND_planet(ccf_indexes, ccf_list_np, ccf_list_np_transpose,
                                            rv_np, s_rv_np, time_np, orbel):
    # perform the L-S fit --------------
    ccf_len = len(ccf_list_np_transpose)
    n_cols = len(ccf_indexes) + 3
    n_rows = len(ccf_list_np)  # len(ccf_list_np)<---- unsmoothed
    x = np.zeros((n_rows, n_cols))

    # Add Keplerian
    x[:, 0] = radvel.kepler.rv_drive(time_np, orbel, use_c_kepler_solver=True)

    # Add a constant term
    x[:, 1] = 1

    # Add ccf indexes
    for i in np.arange(0, len(ccf_indexes)):
        x[:, i + 2] = ccf_list_np_transpose[ccf_indexes[i]]  # (for all observations)

    # add h-alpha
    x[:, n_cols - 1] = ccf_list_np_transpose[ccf_len - 1]
    y = rv_np

    sigma = np.diag(s_rv_np ** 2)
    x_x = np.linalg.inv(sigma).dot(x)
    y_x = np.linalg.inv(sigma).dot(y)

    alpha = x.transpose().dot(x_x)
    beta = x.transpose().dot(y_x)

    # and finally we can write a_coeff = alpha^-1 * beta
    inv_alpha = np.linalg.inv(alpha)
    a_coeff = inv_alpha.dot(beta)

    CCF_matrix = np.zeros((n_rows, len(ccf_indexes)))
    for i in np.arange(0, len(ccf_indexes)):
        CCF_matrix[:, i] = ccf_list_np_transpose[ccf_indexes[i]]  # (for all observations)

    y_preds = x.dot(a_coeff)

    # Compute the scatter metric
    og_rms = new_rms = np.std(y, ddof=1)
    new_rms = np.std(y - y_preds, ddof=1)
    # print("original rms: " + str(round(og_rms, 3)) + ", new rms: " + str(round(new_rms, 3)))

    return a_coeff, y_preds, x, og_rms, new_rms

if __name__ == '__main__':
    Weighted_LS_fit_with_h_alpha_AND_planet()