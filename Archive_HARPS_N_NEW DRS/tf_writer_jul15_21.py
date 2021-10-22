from __future__ import absolute_import, division, print_function
import argparse
import multiprocessing
import os
import sys
import numpy as np
import pdb
from random import shuffle
import pandas as pd
import tensorflow as tf
from astropy.io import fits
from astropy.io.fits import getheader
import tensorflow as tf
from tf_util import example_util

class TfRecordMaker:

    def __init__(self, input_path, path, numfits, index=None):
        self.input_path = input_path or os.input_path.dirname(os.input_path.realpath(__file__))#or 'shifted_fits_clean73_May26_one_file/'  # or os.path.dirname(os.path.realpath(__file__))
        self.path = path or os.path.dirname(os.path.realpath(__file__))
        self.numfits = numfits or 0
        self.index = index

    def make_examples(self):
        examples = []
        print(self.index)
        # index = np.arange(0,self.numfits,1)
        # np.random.seed(42)
        # np.random.shuffle(index)

        rv_non_zero_list = []
        mu_og_fit_list = []
        mu_jup_fit_list = []
        mu_zero_fit_list = []
        n_iters_fit_list = []
        A_fit_list = []
        sigma_fit_list = []
        offset_fit_list = []
        BJD_list = []
        fwhm_list = []
        contrast_list = []
        bis_list = []

        v_true_centered_list = []

        og_ccf_all = []
        jup_shifted_CCF_all = []
        zero_shifted_CCF_all = []
        CCF_norm_all = []
        CCF_residuals_all = []
        CCF_norm_cutoff_all = []
        CCF_residuals_cutoff_all = []

        headr_all = []
        index_number = 0

        # opens data that includes planet shifts, disable this if you open the other files below
        print(self.input_path)
        hdul = fits.open(str(self.input_path) + 'shifted_ccfs_combined.fits', memmap=False)
        # hdul_original = fits.open('Training_set_May23/fits_long_lat_size/ccf' + str(i) + '_PSI=0.000.fits')
        hdul_zero = fits.open(str(self.input_path) + 'shifted_ccfs_combined.fits', memmap=False)

        for i in self.index:
            print("index: " + str(i))

            # extracts activity signal and rv_data
            act_signal = hdul[2].data['vrad_star'][i]

            # if i % 10 == 0:
            #    print(act_signal)

            mu_og_fit = hdul[2].data['mu_og_list'][i]
            mu_jup_fit = hdul[2].data['mu_jup_list'][i]
            mu_zero_fit = hdul[2].data['mu_zero_list'][i]
            BJD = hdul[2].data['bjd'][i]

            fwhm = hdul[2].data['fwhm'][i]
            contrast = hdul[2].data['cont'][i]
            bis = hdul[2].data['bis_span'][i]

            rv_data = hdul[1].data

            if act_signal == 0.0:
                continue

            rv_non_zero_list.append(act_signal)
            mu_og_fit_list.append(mu_og_fit)
            mu_jup_fit_list.append(mu_jup_fit)
            mu_zero_fit_list.append(mu_zero_fit)
            BJD_list.append(BJD)

            fwhm_list.append(fwhm)
            contrast_list.append(contrast)
            bis_list.append(bis)

            # extracts all info from the header
            headr = hdul[1].header.copy()

            # creates arrays where we can store the data to put into the example

            og_ccf_list = hdul[1].data['og_ccf_list'][i]
            jup_shifted_CCF_data_list = hdul[1].data['jup_shifted_CCF_data_list'][i]
            zero_shifted_CCF_list = hdul[1].data['zero_shifted_CCF_list'][i]
            CCF_normalized_list = hdul[1].data['CCF_normalized_list'][i]
            ref_ccf_list = hdul[1].data['CCF_normalized_list'][544]
            CCF_residual_list = CCF_normalized_list - ref_ccf_list

            # add CCF with edges cutoff
            CCF_normalized_list_cutoff = hdul[1].data['CCF_normalized_list_cutoff'][i]
            ref_ccf_list_cutoff = hdul[1].data['CCF_normalized_list_cutoff'][544]
            CCF_residual_list_cutoff = CCF_normalized_list_cutoff - ref_ccf_list_cutoff

            og_ccf_all.append(og_ccf_list)
            jup_shifted_CCF_all.append(jup_shifted_CCF_data_list)
            zero_shifted_CCF_all.append(zero_shifted_CCF_list)
            CCF_norm_all.append(CCF_normalized_list)
            CCF_residuals_all.append(CCF_residual_list)
            headr_all.append(headr)

            CCF_norm_cutoff_all.append(CCF_normalized_list_cutoff)
            CCF_residuals_cutoff_all.append(CCF_residual_list_cutoff)


            hdul.close()
            hdul_zero.close()

        median_rv = np.mean(rv_non_zero_list)
        median_residual = np.median(CCF_residuals_all, axis=0)
        std_residual = np.std(CCF_residuals_all, axis=0)

        median_cutoff_residual = np.median(CCF_residuals_cutoff_all, axis=0)
        std_cutoff_residual = np.std(CCF_residuals_cutoff_all, axis=0)

        for j in np.arange(0, len(CCF_residuals_all)):
            ex = tf.train.Example()

            # Set CCF features.
            example_util.set_float_feature(ex, "OG_CCF",
                                           og_ccf_all[j])
            example_util.set_float_feature(ex, "JUP_CCF",
                                           jup_shifted_CCF_all[j])
            example_util.set_float_feature(ex, "ZERO_CCF",
                                           zero_shifted_CCF_all[j])
            example_util.set_float_feature(ex, "CCF",
                                           CCF_norm_all[j])
            example_util.set_float_feature(ex, "CCF_residuals",
                                           CCF_residuals_all[j])
            example_util.set_float_feature(ex, "Rescaled CCF_residuals",
                                           (CCF_residuals_all[j] - median_residual) / std_residual)
            example_util.set_float_feature(ex, "CCF_cutoff",
                                           CCF_norm_cutoff_all[j])
            example_util.set_float_feature(ex, "CCF_residuals_cutoff",
                                           CCF_residuals_cutoff_all[j])
            example_util.set_float_feature(ex, "Rescaled CCF_residuals_cutoff",
                                           (CCF_residuals_cutoff_all[j] - median_cutoff_residual) / std_cutoff_residual)

            # prints what iteration we are currently at
            index_number = index_number + 1
            if index_number % 500 == 0:
                print(index_number)

            # Set residuals
            # example_util.set_feature(ex, "activity signal residuals", act_signal)
            example_util.set_feature(ex, "activity signal from soap", [(rv_non_zero_list[j])])  # in km/s
            example_util.set_feature(ex, "activity signal", [(rv_non_zero_list[j] - median_rv)])  # in km/s
            example_util.set_feature(ex, "mu_og_fit", [(mu_og_fit_list[j])])
            example_util.set_feature(ex, "mu_jup_fit", [(mu_jup_fit_list[j])])
            example_util.set_feature(ex, "mu_zero_fit", [(mu_zero_fit_list[j])])
            example_util.set_feature(ex, "BJD", [(BJD_list[j])])
            example_util.set_feature(ex, "fwhm", [(fwhm_list[j])])
            example_util.set_feature(ex, "contrast", [(contrast_list[j])])
            example_util.set_feature(ex, "bis", [(bis_list[j])])

            # set the other features in the header
            for k in headr_all[j]:
                example_util.set_feature(ex, str(k), [headr_all[j][k]])

            examples.append(ex)
        return examples

def tf_writer(input_path, path, numfits, randseed):
    num_ccfs = 629
    full_val_cutoff = int(0.80*num_ccfs) # where 628 is the number of nonzero ccfs
    cross_val_cutoff = int(0.08 * num_ccfs)
    val_cutoff = int(0.1*num_ccfs)
    test_cutoff = int(0.1*num_ccfs)
    index = np.arange(0, num_ccfs, 1)
    np.random.seed(randseed)
    np.random.shuffle(index)

    reps_bf = []
    reps_aft = []
    train_indeces = []
    intervals = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80] # fix this (0.08 each) so it's 0.08*10 = 0.80
    for i in range(0, len(intervals)):
        if intervals[i] != 0.08:
            reps_bf.append(int(intervals[i - 1] * num_ccfs))
            reps_aft.append(int(intervals[i] * num_ccfs))
            train_indeces.append(index[int(intervals[i - 1] * num_ccfs):int(intervals[i] * num_ccfs)])
        else:
            print(intervals[i])
            reps_bf.append(0)
            reps_aft.append(int(intervals[i] * num_ccfs))
            train_indeces.append(index[0:int(intervals[i] * num_ccfs)])

    subset0 = train_indeces[1:]
    subset1 = train_indeces[0:1] + train_indeces[2:]
    subset2 = train_indeces[0:2] + train_indeces[3:]
    subset3 = train_indeces[0:3] + train_indeces[4:]
    subset4 = train_indeces[0:4] + train_indeces[5:]
    subset5 = train_indeces[0:5] + train_indeces[6:]
    subset6 = train_indeces[0:6] + train_indeces[7:]
    subset7 = train_indeces[0:7] + train_indeces[8:]
    subset8 = train_indeces[0:8] + train_indeces[9:]
    subset9 = train_indeces[0:9]

    flattened0 = [val for sublist in subset0 for val in sublist]
    flattened1 = [val for sublist in subset1 for val in sublist]
    flattened2 = [val for sublist in subset2 for val in sublist]
    flattened3 = [val for sublist in subset3 for val in sublist]
    flattened4 = [val for sublist in subset4 for val in sublist]
    flattened5 = [val for sublist in subset5 for val in sublist]
    flattened6 = [val for sublist in subset6 for val in sublist]
    flattened7 = [val for sublist in subset7 for val in sublist]
    flattened8 = [val for sublist in subset8 for val in sublist]
    flattened9 = [val for sublist in subset9 for val in sublist]
    indexes_full_val = [val for sublist in train_indeces for val in sublist]

    full_train_flats = []
    full_train_flats.extend([flattened0, flattened1, flattened2, flattened3, flattened4, flattened5, flattened6, flattened7, flattened8, flattened9])

    #train_index = index[0:train_cutoff]
    val_index = index[int(0.8 * num_ccfs):int(0.9 * num_ccfs)]
    test_index = index[int(0.9 * num_ccfs):]

    # # loop through cross_val sets
    # for iteration in range(0, len(full_train_flats)):
    #     with tf.python_io.TFRecordWriter('Archive_HARPS_N/TF_record_Jul_8/TF_ccf_train'+str(iteration)) as writer:
    #         tf_record_maker = TfRecordMaker(path=path, numfits=numfits, index=np.array(full_train_flats[iteration]))
    #         number_examples_train = 0
    #         examples_tf = tf_record_maker.make_examples()
    #         for example in examples_tf[0:train_cutoff]:
    #             print("train")
    #             number_examples_train = number_examples_train + 1
    #             if number_examples_train % 100 == 0:
    #                 print("iteration for training set: " + str(number_examples_train))
    #             writer.write(example.SerializeToString())
    #             # print(ex)

    # Make directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    for iteration in range(0, len(train_indeces)):
        with tf.io.TFRecordWriter(path+'TF_ccf_cross_val'+str(iteration)) as writer:
             tf_record_maker = TfRecordMaker(input_path=input_path, path=path, numfits=numfits, index=train_indeces[iteration])
             number_examples_val = 0
             eval_counter = 0
             #for example in tf_record_maker.make_examples()[train_cutoff:val_cutoff]:
             for example in tf_record_maker.make_examples()[0:cross_val_cutoff+1]:
                 eval_counter += 1
                 print("val: " + str(eval_counter))
                 number_examples_val = number_examples_val + 1
                 if number_examples_val%100 == 0:
                     print("iteration for evaluation set: "+str(number_examples_val))
                 writer.write(example.SerializeToString())
                 # print(ex)

    with tf.io.TFRecordWriter(path+'TF_ccf_val') as writer:
        tf_record_maker = TfRecordMaker(input_path=input_path, path=path, numfits=numfits, index=val_index)
        number_examples_test = 0
        test_counter = 0
        for example in tf_record_maker.make_examples()[0:val_cutoff+1]:
        #for example in tf_record_maker.make_examples()[val_cutoff:]:
            test_counter += 1
            print("test: " + str(test_counter))
            number_examples_test = number_examples_test + 1
            if number_examples_test % 100 == 0:
                print("iteration for testing set: "+str(number_examples_test))
            writer.write(example.SerializeToString())
            # print(ex)


    with tf.io.TFRecordWriter(path+'TF_ccf_test') as writer:
        tf_record_maker = TfRecordMaker(input_path=input_path, path=path, numfits=numfits, index=test_index)
        number_examples_test = 0
        test_counter = 0
        for example in tf_record_maker.make_examples()[0:test_cutoff+1]:
        #for example in tf_record_maker.make_examples()[val_cutoff:]:
            test_counter += 1
            print("test: " + str(test_counter))
            number_examples_test = number_examples_test + 1
            if number_examples_test % 100 == 0:
                print("iteration for testing set: "+str(number_examples_test))
            writer.write(example.SerializeToString())
            # print(ex)

    # Optional: also write a file with all the evaluation files in one file
    #full_val_cutoff
    #indexes_full_val

    with tf.io.TFRecordWriter(path+'TF_ccf_full_train') as writer:
        tf_record_maker = TfRecordMaker(input_path=input_path, path=path, numfits=numfits, index=indexes_full_val)
        number_examples_val = 0
        eval_counter = 0
        # for example in tf_record_maker.make_examples()[train_cutoff:val_cutoff]:
        for example in tf_record_maker.make_examples()[0:full_val_cutoff+1]:
            eval_counter += 1
            print("val: " + str(eval_counter))
            number_examples_val = number_examples_val + 1
            if number_examples_val % 100 == 0:
                print("iteration for evaluation set: " + str(number_examples_val))
            writer.write(example.SerializeToString())
            # print(ex)