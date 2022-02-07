# import standard packages
import os.path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import trange

# rv_net imports.
import sys
sys.path.append("rv_net/")
from ops import training
from tf_util import config_util
from tf_util import configdict
from tf_util import estimator_runner
from rv_net import data_HARPS_N
from rv_net import  data, rv_model, estimator_util, load_dataset_ridge, ridge_regress_harps

# Read in the Data
eval_method ='cross_val' #"val"#'cross_val' #"val"# "val"#"cross_val" # "val" # "test"
ccf_len = 46

# Read in files for cross-validation
DATA_DIR = 'TF_records_Nov2021/' #TF_record_July_10_21_no_planets_median_prov_rvs/' #HARPS-N Solar Telescope Data (using old DRS)/' #TF_record_July_10_21_no_planets_same_test_set' #TF_record_July_10_21_no_planets_v2'

if eval_method =="cross_val":
  data_files = tf.data.Dataset.list_files(DATA_DIR+'*cross_val*',shuffle=False)
  data_files = [t.numpy() for t in data_files]
  TRAIN_FILE_NAME_LIST = []
  VAL_FILE_NAME_LIST = []

  N = len(data_files)
  for i in range(N):
    val_files = [data_files[i]]
    #print(val_files)
    VAL_FILE_NAME_LIST.append(val_files)
    train_files = data_files[0:i] + data_files[i+1:]
    TRAIN_FILE_NAME_LIST.append(train_files)
    # add all the training files

  NUM_TRAINING_EXAMPLES = 503
  NUM_VALIDATION_EXAMPLES = 51

elif eval_method =="val":
  TRAIN_FILE_NAME_LIST = [[os.path.join(DATA_DIR, "TF_ccf_full_train")]]
  VAL_FILE_NAME_LIST = [[os.path.join(DATA_DIR, "TF_ccf_val")]]#test")]]

  NUM_TRAINING_EXAMPLES = 503
  NUM_VALIDATION_EXAMPLES = 61
elif eval_method =="test":
  TRAIN_FILE_NAME_LIST = [[os.path.join(DATA_DIR, "TF_ccf_full_train")]]
  VAL_FILE_NAME_LIST = [[os.path.join(DATA_DIR, "TF_ccf_test")]]#test")]]

  NUM_TRAINING_EXAMPLES = 503
  NUM_VALIDATION_EXAMPLES = 61
else:
  print("Please select a valid evaluation method: 'cross_val' or 'val' or 'test'")

def _example_parser(serialized_example):
    """Parses a single tf.Example into feature and label tensors."""
    feature_name = "Rescaled CCF_residuals_cutoff"#"Rescaled CCF_residuals" #CCF_residuals
    label_name = "activity signal"#"RV",
    label2_name = "BJD"
    data_fields = {
        feature_name: tf.io.FixedLenFeature([ccf_len], tf.float32), #[161], tf.float32),
        label_name: tf.io.FixedLenFeature([], tf.float32),
        label2_name: tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_fields = tf.io.parse_single_example(serialized_example, features=data_fields)
    return parsed_fields[feature_name], parsed_fields[label_name]*1000, parsed_fields[label2_name]

def load_dataset(filenames, batch_size, mode=tf.estimator.ModeKeys.EVAL):
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = filename_dataset.flat_map(tf.data.TFRecordDataset)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=NUM_TRAINING_EXAMPLES)
    dataset = dataset.map(_example_parser, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset


# @title Define LinearModel, FCModel, CNNModel
class LinearModel(tf.keras.Model):
    """A TensorFlow linear regression model."""

    def __init__(self, hparams):
        """Basic setup.

        Args:
          hparams: A ConfigDict of hyperparameters for building the model.

        Raises:
          ValueError: If mode is invalid.
        """
        super(LinearModel, self).__init__()
        self.hparams = hparams
        # self.weights = tf.Variable(tf.zeros(self.hparams.num_features))
        self.dense_layer = tf.keras.layers.Dense(
            1, kernel_initializer=tf.zeros_initializer, use_bias=False)

    def call(self, features, training=False):
        # return tf.tensordot(features, self.weights, axes=1)
        return tf.squeeze(self.dense_layer(features))


class FCModel(tf.keras.Model):
    """A TensorFlow linear regression model."""

    def __init__(self, hparams):
        """Basic setup.

        Args:
          hparams: A ConfigDict of hyperparameters for building the model.

        Raises:
          ValueError: If mode is invalid.
        """
        super(FCModel, self).__init__()
        self.hparams = hparams
        # self.hidden_layer1 = tf.keras.layers.Dense(
        #    self.hparams.num_dense_units, activation=tf.keras.activations.relu)
        self.dense_layers = [
            tf.keras.layers.Dense(
                hparams.num_dense_units,
                activation=tf.keras.activations.relu)
            for i in range(hparams.num_dense_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, features, training=False):
        net = tf.expand_dims(features, -1)
        batch_size, length, depth = net.shape
        net = tf.reshape(net, [batch_size, length * depth])
        for dense in self.dense_layers:
            net = dense(net)
        net = self.output_layer(net)
        return tf.squeeze(net)


# @title Define RVLinearModel
class CNNModel(tf.keras.Model):
    """A TensorFlow linear regression model."""

    def __init__(self, hparams):
        """Basic setup.

        Args:
          hparams: A ConfigDict of hyperparameters for building the model.

        Raises:
          ValueError: If mode is invalid.
        """
        super(CNNModel, self).__init__()
        self.hparams = hparams
        self.conv_layers = [
            tf.keras.layers.Conv1D(
                filters=hparams.num_conv_filters,
                kernel_size=hparams.conv_kernel_size,
                activation=tf.keras.activations.relu,
                padding="same")
            for i in range(hparams.num_conv_layers)
        ]
        self.dense_layers = [
            tf.keras.layers.Dense(
                hparams.num_dense_units,
                activation=tf.keras.activations.relu)
            for i in range(hparams.num_dense_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, features, training=False):
        net = tf.expand_dims(features, -1)
        for conv in self.conv_layers:
            net = conv(net)
        batch_size, length, depth = net.shape
        net = tf.reshape(net, [batch_size, length * depth])
        for dense in self.dense_layers:
            net = dense(net)
        net = self.output_layer(net)
        return tf.squeeze(net)


def make_predictions(model, dataset):
    all_preds = []
    all_labels = []
    all_bjds = []
    for features, labels, bjds in dataset:
        preds = model(features, training=False)
        all_preds.append(preds.numpy())
        all_labels.append(labels.numpy())
        all_bjds.append(bjds.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_bjds)


def train(model, hparams, plots, model_name, num_epochs):  # =100):
    train_dataset = load_dataset([TRAIN_FILE_NAME], batch_size=hparams.batch_size, mode=tf.estimator.ModeKeys.TRAIN)
    val_dataset = load_dataset([VAL_FILE_NAME], batch_size=min(1024, NUM_VALIDATION_EXAMPLES),
                               mode=tf.estimator.ModeKeys.EVAL)
    loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    Opt = tfa.optimizers.extend_with_decoupled_weight_decay(tf.optimizers.SGD)
    optimizer = Opt(weight_decay=hparams.weight_decay, learning_rate=hparams.learning_rate, momentum=hparams.momentum)
    metrics = [
        tf.keras.metrics.MeanSquaredError("train_loss"),
        tf.keras.metrics.RootMeanSquaredError("train_rmse")
    ]
    weight_decay_list_t.append(hparams.weight_decay)
    gaussian_noise_list_t.append(hparams.gaussian_noise_scale)

    metric_values = []
    for epoch in range(1, num_epochs + 1):
        # Reset metric values for each new epoch.
        for m in metrics:
            m.reset_states()

        # Train over all batches in the training set.
        for features, labels, bjds in train_dataset:
            if hparams.gaussian_noise_scale:
                features += tf.random.normal(features.shape, stddev=hparams.gaussian_noise_scale)
                # print(hparams.gaussian_noise_scale)
            # One training step.
            with tf.GradientTape() as t:
                preds = model(features, training=True)
                loss = loss_fn(labels, preds)
            grads = t.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Update the metrics.
            for m in metrics:
                m(labels, preds)

        # End of an epoch.
        epoch_metrics = {"epoch": epoch}
        # First, log the training metrics.
        for m in metrics:
            epoch_metrics[m.name] = m.result().numpy()
        # Next, evaluate over the validation set.
        labels_val, preds_val, bjd_val = make_predictions(model, val_dataset)
        epoch_metrics["val_rmse"] = np.sqrt(np.mean(np.square(preds_val - labels_val)))
        # Add a metric for the raw scatter started with
        epoch_metrics["original_rmse"] = np.std(labels_val)
        # Add a metric for raw scatter - corrected scatter
        epoch_metrics["difference_rmse"] = np.std(labels_val) - np.sqrt(np.mean(np.square(preds_val - labels_val)))
        # Log metrics to tensorboard.
        for metric, value in epoch_metrics.items():
            tf.summary.scalar(metric, value, step=epoch)
        epoch_metrics["epoch"] = epoch
        # Print metric values at selected epochs.
        #if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
        #    print("{epoch}: Train loss: {train_loss:.4}, Train RMSE: {train_rmse:.4}, Val RMSE: {val_rmse:.4}".format(
        #        **epoch_metrics))
        metric_values.append(epoch_metrics)

    # Gather predictions
    labels, preds, bjd = make_predictions(model, train_dataset)
    labels_val, preds_val, bjd_val = make_predictions(model, val_dataset)
    all_bjds_val.append(bjd_val)
    bjd_run_val.append(bjd_val)
    all_pred_val.append(preds_val)
    pred_run_val.append(preds_val)
    all_labels_val.append(labels_val)
    labels_run_val.append(labels_val)

    # Scatter reduction plot
    sd_x = np.std(labels_val, ddof=1)
    rms_x = np.sqrt(np.mean(np.square(labels_val - preds_val)))
    rms_x_list.append(rms_x)
    rms_avg_list.append(rms_x)
    stel_removed = np.sqrt(np.abs(sd_x ** 2 - rms_x ** 2))
    x_range = np.linspace(-4, 5.5, 17)
    upper_bound = x_range + rms_x
    lower_bound = x_range - rms_x

    if plots == "ON":
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        ax = axes[0]
        ax.plot([m["train_rmse"] for m in metric_values], label="Train RMSE")
        ax.plot([m["val_rmse"] for m in metric_values], label="Validation RMSE")
        ax.set_xlabel("Epoch")
        ax.legend(loc="upper right")

        # Gather predictions to plot against labels.
        ax = axes[1]
        ax.plot(preds, labels, ".", label="Training")
        ax.plot(preds_val, labels_val, ".", label="Validation")
        ax.set_xlabel("Actual Y")
        ax.set_ylabel("Predicted Y")
        ax.legend(loc="lower right")

        # plot the scatter reduction plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(labels_val, preds_val, ".")
        ax1.plot(x_range, x_range, color="blue", label="1:1 ratio")
        # ax.plot(x_range,z[0]*x_range+z[1], color="blue")
        rms_fill = rms_x  # 0.15
        ax1.fill_between(x_range, x_range + rms_fill, x_range - rms_fill, facecolor='lightblue',
                         alpha=0.5, label="1 standard deviation")
        ax1.set_xlim(-4, 4);
        ax1.set_ylim(-4, 4);
        ax1.set_xlabel("HARPS-N Stellar Activity Signal (m/s)", size=16)
        ax1.set_ylabel("Model Predicted Stellar Activity Signal (m/s)", size=16)
        ax1.set_title(
            model_name + " Model Predictions of Stellar Activity signal(m/s)")  # , %d epochs, weight decay: %.2e, gauss noise: %.2e " %(num_epochs,
        # hparams.weight_decay, hparams.gaussian_noise_scale, size=16)
        textstr = '\n'.join((
            r'Raw scatter=%.3f m/s' % (sd_x,),
            r'Corrected scatter=%.3f m/s' % (rms_x,),
            r'Stellar Error Removed=%.3f m/s' % (stel_removed,)))
        ax1.text(-3.8, 3.5, textstr, size=15,
                 ha="left", va="top",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           ))
        ax1.legend(loc="lower right")
    else:
        textstr = '\n'.join((
            r'Raw scatter=%.3f m/s' % (sd_x,),
            r'Corrected scatter=%.3f m/s' % (rms_x,),
            r'Stellar Error Removed=%.3f m/s' % (stel_removed,)))
        #print(textstr)

    return metric_values

## Hyperparameter optimization
# Define the ranges of values that each hyperparameter can take

exp_list = np.random.uniform(2,4, 1000)
learning_rate_list = [10**-exp_list[0], 10**-exp_list[1], 10**-exp_list[2]]

conv_kernel_size_list = [3,5,7]
num_conv_filters_list = [8, 16, 32]
num_conv_layers_list = [2, 4, 6]
dense_units_list = [100, 200, 500, 1000]
num_dense_layers_list = [1, 2, 4, 6, 8]
weight_decay_list = np.geomspace(0.0005, 0.05, 1000)
num_epochs_list = [50, 55, 65, 70, 80, 90, 100]


LOG_DIR_4 = "CNN_tensorlogs/" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
#print(LOG_DIR_4)

# Log the hparam config. This is optional but makes tensorboard show the right ranges.
with tf.summary.create_file_writer(LOG_DIR_4).as_default():
  hp.hparams_config(
    hparams=[
             hp.HParam('learning_rate', hp.RealInterval(1e-4, 1e-2)),
             hp.HParam('conv_kernel_size', hp.Discrete([3, 5, 7])),
             hp.HParam('num_conv_filters', hp.Discrete([8, 16, 32])),
             hp.HParam('num_conv_layers', hp.Discrete([2, 4, 6])),
             hp.HParam('num_dense_units', hp.Discrete([100, 200, 500, 1000])),
             hp.HParam('num_dense_layers', hp.Discrete([1, 2, 4, 6, 8])),
             hp.HParam('weight_decay', hp.RealInterval(5e-4,5e-2)),
             hp.HParam('epochs', hp.Discrete([50, 55, 65, 70, 80, 90, 100]))],
    metrics=[hp.Metric("val_rmse", display_name='RMSE'),
             hp.Metric("original_rmse", display_name="OG RMSE"),
             hp.Metric("difference_rmse", display_name="diff RMSE")
             ],
  )

model_num = 1
RMSE_list = []
all_pred_bjd_val = []

for i in trange(0,2):#30):
      rms_avg_list = []
      weight_decay_list_t = []
      gaussian_noise_list_t = []
      rms_x_list = []
      all_bjds_val = []
      all_pred_val = []
      all_labels_val = []
      all_mean_val_preds = []
      all_mean_val_bjds = []
      all_mean_val_labels = []

      hparams = configdict.ConfigDict(dict(
          num_features=46,
          learning_rate=np.random.choice(learning_rate_list),
          momentum=0.9,
          batch_size=1024,
          conv_kernel_size=int(np.random.choice(conv_kernel_size_list)),
          num_conv_filters=int(np.random.choice(num_conv_filters_list)),
          num_conv_layers=int(np.random.choice(num_conv_layers_list)),
          num_dense_units=int(np.random.choice(dense_units_list)),
          num_dense_layers=int(np.random.choice(num_dense_layers_list)),
          weight_decay=np.random.choice(weight_decay_list), #5e-4, #7e-2,
          gaussian_noise_scale=0,#1.5,
          epochs = int(np.random.choice(num_epochs_list))
      ))
      epochs =  hparams.epochs
      for index in range(0, len(VAL_FILE_NAME_LIST)):
        TRAIN_FILE_NAME = TRAIN_FILE_NAME_LIST[index]
        VAL_FILE_NAME = VAL_FILE_NAME_LIST[index]

        bjd_run_val = []
        pred_run_val = []
        labels_run_val = []
        #print("Model {}. Learning_rate: {:.5f}. Num conv filters: {}. Num conv layers: {}. Dense units: {}. Dense layers {}. Weight decay: {:.5f}".format(
        #    model_num, hparams.learning_rate, hparams.num_conv_filters, hparams.num_conv_layers, hparams.num_dense_units, hparams.num_dense_layers, hparams.weight_decay))
        #print("Cross-val number: "+str(index+1)+", Model archictect. num: "+str(i+1)+", Run number: "+str(k+1))
        model = CNNModel(hparams)
        run_dir = LOG_DIR_4 + "/{}/".format(model_num)
        with tf.summary.create_file_writer(run_dir).as_default():
          hp.hparams(hparams)  # record the values used in this trial
          metric_values = train(model, hparams, plots="OFF", model_name = "CNN",num_epochs=epochs)#np.random.choice(num_epochs_list))
          final_metrics = metric_values[-1]

        #print("________________________")
        #print()
        mean_val_preds = np.mean(pred_run_val, axis=0)
        mean_val_labels = np.mean(labels_run_val, axis=0)
        mean_val_bjds = np.mean(bjd_run_val, axis=0)

        all_mean_val_preds.append(mean_val_preds)
        all_mean_val_labels.append(mean_val_labels)
        all_mean_val_bjds.append(mean_val_bjds)

        all_pred_val.append(pred_run_val)
        all_labels_val.append(labels_run_val)
        all_bjds_val.append(bjd_run_val)
      model_num += 1