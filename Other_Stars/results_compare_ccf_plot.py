import numpy as np
import math as m

import matplotlib.pyplot as plt
import matplotlib as mpl


def results_compare_ccf_plot(bjd, y, y_preds, raw_rms, ccf_corr_rms,
                         y_preds_hsb, ccf_corr_rms_hsb, star_name):
    fig, ax = plt.subplots(2,2, figsize=(14, 10))
    #first plot the ccf-only results
    ax[0][0].scatter(bjd,
                  y, label="raw rvs, rms scatter:"+str(round(raw_rms,3)))
    ax[0][0].scatter(bjd,
                  y_preds, label="CCF predictions, rms scatter: "+str(round(ccf_corr_rms,3)))
    ax[0][0].set_ylabel("RV (m/s)")
    ax[0][0].set_xlabel("BJD")
    ax[0][0].set_title("Binned LS ccf only fit: "+star_name)
    ax[0][0].legend()

    ax[1][0].scatter(y, y_preds)
    ax[1][0].plot([-6,6],[-6,6])
    ax[1][0].set_xlabel("measured RV")
    ax[1][0].set_ylabel("model predicted RV")
    ax[1][0].set_title("Binned LS ccf only fit: "+star_name)

    #Now, also plot the ccf+s-index+h-alpha results
    ax[0][1].scatter(bjd,
                  y, label="raw rvs, rms scatter:"+str(round(raw_rms,3)))
    ax[0][1].scatter(bjd,
                  y_preds_hsb, label="CCF predictions, rms scatter: "+str(round(ccf_corr_rms_hsb,3)))
    ax[0][1].set_ylabel("RV (m/s)")
    ax[0][1].set_xlabel("BJD")
    ax[0][1].set_title("Binned LS ccf, h-alpha fit: "+star_name)
    ax[0][1].legend()

    ax[1][1].scatter(y, y_preds_hsb)
    ax[1][1].plot([-6,6],[-6,6])
    ax[1][1].set_xlabel("measured RV")
    ax[1][1].set_ylabel("model predicted RV")
    ax[1][1].set_title("Binned LS ccf, h-alpha fit: "+star_name)


if __name__ == '__main__':
    results_compare_ccf_plot()