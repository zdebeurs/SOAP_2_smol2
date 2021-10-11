import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl
from Weighted_LS_fit import Weighted_LS_fit


# Compute Bayesian Information Criterion
def BIC(ccf_indexes, ccf_list_np, ccf_list_np_transpose, rv_np, s_rv_np):
    a_coeff, y_preds, x, og_rms, new_rms = Weighted_LS_fit(ccf_indexes,
                                                           ccf_list_np,
                                                           ccf_list_np_transpose,
                                                           rv_np, s_rv_np)

    def log_likelihood(theta, x, y, e):
        a = theta
        model = x.dot(a)
        sigma2 = e ** 2
        loglikelihood = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return loglikelihood

    LS_log_likelihood = log_likelihood(a_coeff, x, rv_np, s_rv_np)
    BIC = -2 * LS_log_likelihood + np.log(len(rv_np)) * len(a_coeff)

    return BIC

if __name__ == '__main__':
    BIC()