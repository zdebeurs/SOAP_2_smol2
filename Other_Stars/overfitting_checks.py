import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl

from Weighted_LS_fit import Weighted_LS_fit

def overfitting_checks(num_runs, ccf_indexes, ccf_list_np, ccf_list_np_transpose, rv_np, s_rv_np):
    fig, ax = plt.subplots(1,2, figsize=(14, 5))

    # Here we check how the rms changes when ALL indexes are shifted by a x
    ccf_indexes_list1 = []
    delta_full_ind_list = []
    ccf_corr_rms_list1 = []
    num_runs1=50

    for i in range(0, num_runs):
        delta_ind = np.random.choice(np.arange(-8,8))
        delta_full_ind_list.append(delta_ind)
        ccf_indexes = ccf_indexes.copy()+delta_ind


        a_coeff, y_preds, x, og_rms, new_rms = Weighted_LS_fit(ccf_indexes, ccf_list_np,ccf_list_np_transpose, rv_np, s_rv_np)

        ccf_indexes_list1.append(ccf_indexes)
        ccf_corr_rms_list1.append(new_rms)

        #print("-----------------")
        #print("run: "+str(i)+", indexes: "+str(ccf_indexes))
        #print("raw rvs, rms scatter:"+str(round(og_rms,3)))
        #print("CCF predictions, rms scatter: "+str(round(new_rms,3)))
        ccf_indexes = ccf_indexes.copy()

    ax[0].scatter(delta_full_ind_list, ccf_corr_rms_list1)
    ax[0].set_xlabel("index shift for ALL indexes")
    ax[0].set_ylabel("corrected rms scatter")


    # Next, we check how the rms changes when ONE random index is shifted by x
    ccf_indexes_list = []
    delta_ind_list = []
    ccf_corr_rms_list = []
    num_runs=50

    for i in range(0, num_runs):
        rand_index = np.random.randint(0, len(ccf_indexes))
        delta_ind = np.random.choice(np.arange(-8,8))

        #check whether this index is already an existing index
        while ccf_indexes[rand_index]+delta_ind in ccf_indexes:
            rand_index = np.random.randint(0, len(ccf_indexes))
            delta_ind = np.random.choice(np.arange(-8,8))

        delta_ind_list.append(delta_ind)
        ccf_indexes[rand_index] = ccf_indexes[rand_index].copy()+delta_ind


        a_coeff, y_preds, x, og_rms, new_rms = Weighted_LS_fit(ccf_indexes, ccf_list_np,ccf_list_np_transpose, rv_np, s_rv_np)

        ccf_indexes_list.append(ccf_indexes)
        ccf_corr_rms_list.append(new_rms)

        #print("-----------------")
        #print("run: "+str(i)+", indexes: "+str(ccf_indexes))
        #print("raw rvs, rms scatter:"+str(round(og_rms,3)))
        #print("CCF predictions, rms scatter: "+str(round(new_rms,3)))
        ccf_indexes = ccf_indexes.copy()

    print(len(delta_ind_list))
    print(len(ccf_corr_rms_list))
    ax[1].scatter(delta_ind_list, ccf_corr_rms_list)
    ax[1].set_xlabel("index shift for ONE index at a time")
    ax[1].set_ylabel("corrected rms scatter")


if __name__ == '__main__':
    overfitting_checks()