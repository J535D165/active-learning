# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run active learner on classification tasks.

Supported datasets include mnist, letter, cifar10, newsgroup20, rcv1,
wikipedia attack, and select classification datasets from mldata.
See utils/create_data.py for all available datasets.

For binary classification, mnist_4_9 indicates mnist filtered down to just 4
and 9. By default uses logistic regression but can also train using kernel
SVM. 2 fold cv is used to tune regularization parameter over a exponential
grid.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import argparse
from time import gmtime
from time import strftime

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from tensorflow import gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils

# initialise the arguments
parser = argparse.ArgumentParser(description='Active learning parameters')

parser.add_argument(
    "--dataset", type=str, default="letter", help="Dataset name")
parser.add_argument(
    "--sampling_method",
    default="margin",
    type=str,
    help="Name of sampling method to use, can be any defined in "
    "AL_MAPPING in sampling_methods.constants")
parser.add_argument(
    "--warmstart_size",
    default=0.02,
    type=float,
    help="Can be float or integer. Float indicates percentage of training data "
    "to use in the initial warmstart model")
parser.add_argument(
    "--batch_size",
    default=0.02,
    type=float,
    help="Can be float or integer. Float indicates batch size as a percentage "
    "of training data size.")
parser.add_argument(
    "--trials",
    default=1,
    type=int,
    help="Number of curves to create using different seeds")
parser.add_argument(
    "--seed", default=1, type=int, help="Seed to use for rng and random state")
parser.add_argument(
    "--confusions",
    default="0.",
    type=str,
    help="Percentage of labels to randomize")
parser.add_argument(
    "--active_sampling_percentage",
    default="1.0",
    type=str,
    help="Mixture weights on active sampling.")
parser.add_argument(
    "--score_method",
    default="logistic",
    type=str,
    help="Method to use to calculate accuracy.")
parser.add_argument(
    "--select_method",
    default="None",
    type=str,
    help="Method to use for selecting points.")
parser.add_argument(
    "--normalize_data",
    default=False,
    type=bool,
    help="Whether to normalize the data.")
parser.add_argument(
    "--standardize_data",
    default=True,
    type=bool,
    help="Whether to standardize the data.")
parser.add_argument(
    "--save_dir",
    default="/tmp/toy_experiments",
    type=str,
    help="Where to save outputs")
parser.add_argument(
    "--data_dir",
    default="/tmp/data",
    type=str,
    help="Directory with predownloaded and saved datasets.")
parser.add_argument(
    "--max_dataset_size",
    default=15000,
    type=int,
    help="maximum number of datapoints to include in data "
    "zero indicates no limit")
parser.add_argument(
    "--train_horizon",
    default=1.0,
    type=float,
    help="how far to extend learning curve as a percent of train")
parser.add_argument(
    "--do_save",
    default=True,
    type=bool,
    help="whether to save log and results")

get_wrapper_AL_mapping()


def generate_one_curve(X,
                       y,
                       sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       select_model=None,
                       confusion=0.,
                       active_p=1.0,
                       max_points=None,
                       standardize_data=False,
                       norm_data=False,
                       train_horizon=0.5):
    """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float or int.  float indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    select_model: defaults to None, in which case the score model will be
      used to select new datapoints to label.  Model must implement fit,
      predict and depending on AL method may also need decision_function.
    confusion: percentage of labels of one class to flip to the other
    active_p: percent of batch to allocate to active learning
    max_points: limit dataset size for preliminary
    standardize_data: wheter to standardize the data to 0 mean unit variance
    norm_data: whether to normalize the data.  Default is False for logistic
      regression.
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """

    # TODO(lishal): add option to find best hyperparameter setting first on
    # full dataset and fix the hyperparameter for the rest of the routine
    # This will save computation and also lead to more stable behavior for the
    # test accuracy

    # TODO(lishal): remove mixture parameter and have the mixture be specified
    # as a mixture of samplers strategy
    def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                     **kwargs):
        n_active = int(mixture * N)
        n_passive = N - n_active
        kwargs["N"] = n_active
        kwargs["already_selected"] = already_selected
        batch_AL = sampler.select_batch(**kwargs)
        already_selected = already_selected + batch_AL
        kwargs["N"] = n_passive
        kwargs["already_selected"] = already_selected
        batch_PL = uniform_sampler.select_batch(**kwargs)
        return batch_AL + batch_PL

    # set a random seed
    # is this a correct way to do this?
    np.random.seed(seed)
    data_splits = [2. / 3, 1. / 6, 1. / 6]

    # 2/3 of data for training
    if max_points is None:
        max_points = len(y)
    train_size = int(min(max_points, len(y)) * data_splits[0])

    # Compute the batch size if it is less than 1. Then it is the batch_size
    # multiplied by the train_size
    if batch_size < 1:
        batch_size = batch_size * train_size
    batch_size = int(batch_size)

    # Use a warm start.
    if warmstart_size < 1:
        # Set seed batch to provide enough samples to get at least 4 per class
        # TODO(lishal): switch to sklearn stratified sampler
        seed_batch = int(warmstart_size * train_size)
    else:
        seed_batch = int(warmstart_size)
    seed_batch = max(seed_batch, 6 * len(np.unique(y)))

    # make a split of the data: switch to sklearn data splitter?
    indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
        utils.get_train_val_test_splits(
            X, y, max_points, seed, confusion, seed_batch, split=data_splits))

    # Preprocess data
    if norm_data:
        print("Normalizing data")
        X_train = normalize(X_train)
        X_val = normalize(X_val)
        X_test = normalize(X_test)
    if standardize_data:
        print("Standardizing data")
        scaler = StandardScaler(with_mean=False).fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    print("active percentage: {} warmstart batch: {} "
          "batch size: {} confusion: {} seed: {}".format(
              active_p, seed_batch, batch_size, confusion, seed))

    # Initialize samplers
    uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
    sampler = sampler(X_train, y_train, seed)

    results = {}
    data_sizes = []
    accuracy = []
    selected_inds = list(range(seed_batch))

    # If select model is None, use score_model
    same_score_select = False
    if select_model is None:
        select_model = score_model
        same_score_select = True

    n_batches = int(
        np.ceil(
            (train_horizon * train_size - seed_batch) * 1.0 / batch_size)) + 1
    for b in range(n_batches):
        n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
        print("Training model on " + str(n_train) + " datapoints")

        assert n_train == len(selected_inds)
        data_sizes.append(n_train)

        # Sort active_ind so that the end results matches that of uniform
        # sampling
        partial_X = X_train[sorted(selected_inds)]
        partial_y = y_train[sorted(selected_inds)]
        score_model.fit(partial_X, partial_y)
        if not same_score_select:
            select_model.fit(partial_X, partial_y)
        acc = score_model.score(X_test, y_test)
        accuracy.append(acc)
        print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name,
                                                 accuracy[-1] * 100))

        n_sample = min(batch_size, train_size - len(selected_inds))
        select_batch_inputs = {
            "model": select_model,
            "labeled": dict(zip(selected_inds, y_train[selected_inds])),
            "eval_acc": accuracy[-1],
            "X_test": X_val,
            "y_test": y_val,
            "y": y_train
        }
        new_batch = select_batch(sampler, uniform_sampler, active_p, n_sample,
                                 selected_inds, **select_batch_inputs)
        selected_inds.extend(new_batch)

        # it seems that a difference between the requested and selected
        # samples is possible. mayby in case of already reviewed samples.
        print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
        assert len(new_batch) == n_sample  # raises if not equal
        assert len(list(set(selected_inds))) == len(selected_inds)

    # Check that the returned indice are correct and will allow mapping to
    # training set from original data
    assert all(y_noise[indices[selected_inds]] == y_train[selected_inds])
    results["accuracy"] = accuracy
    results["selected_inds"] = selected_inds
    results["data_sizes"] = data_sizes
    results["indices"] = indices
    results["noisy_targets"] = y_noise
    return results, sampler


def main(args):

    # make the export folder structure
    # this is made here because the Logger uses the filename
    if args.do_save:
        # make a base save directory
        utils.make_dir(args.save_dir)

        # make a directory in the base save directory with for the specific
        # method.
        save_subdir = os.path.join(args.save_dir,
                                   args.dataset + "_" + args.sampling_method)
        utils.make_dir(save_subdir)

        filename = os.path.join(
            save_subdir,
            "log-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt")
        sys.stdout = utils.Logger(filename)

    # confusion argument can have multiple values
    confusions = [float(t) for t in args.confusions.split(" ")]
    mixtures = [float(t) for t in args.active_sampling_percentage.split(" ")]
    max_dataset_size = None if args.max_dataset_size == 0 else args.max_dataset_size
    starting_seed = args.seed

    # get the dataset from file based on the data directory and dataset name
    X, y = utils.get_mldata(args.data_dir, args.dataset)

    # object to store the results in
    all_results = {}

    # percentage of labels to randomize
    for c in confusions:

        # Mixture weights on active sampling."
        for m in mixtures:

            # the number of curves created during multiple trials
            for seed in range(starting_seed, starting_seed + args.trials):

                # get the sampler based on the name
                # returns a python object
                # also named: query strategy
                sampler = get_AL_sampler(args.sampling_method)

                # get the model
                score_model = utils.get_model(args.score_method, seed)

                #
                if (args.select_method == "None"
                        or args.select_method == args.score_method):
                    select_model = None
                else:
                    select_model = utils.get_model(args.select_method, seed)

                # create the learning curve
                results, sampler_state = generate_one_curve(
                    X,
                    y,
                    sampler,
                    score_model,
                    seed,
                    args.warmstart_size,
                    args.batch_size,
                    select_model,
                    confusion=c,
                    active_p=m,
                    max_points=max_dataset_size,
                    standardize_data=args.standardize_data,
                    norm_data=args.normalize_data,
                    train_horizon=args.train_horizon)
                key = (args.dataset, args.sampling_method, args.score_method,
                       args.select_method, m, args.warmstart_size,
                       args.batch_size, c, args.standardize_data,
                       args.normalize_data, seed)
                sampler_output = sampler_state.to_dict()
                results["sampler_output"] = sampler_output
                all_results[key] = results

    # Not sure why this is done in a qay like this.
    fields = [
        "dataset", "sampler", "score_method", "select_method",
        "active percentage", "warmstart size", "batch size", "confusion",
        "standardize", "normalize", "seed"
    ]
    all_results["tuple_keys"] = fields

    # write the results to a file
    if args.do_save:

        # format the filename
        filename = "results_score_{}_select_{}_norm_{}_stand_{}".format(
            args.score_method, args.select_method, args.normalize_data,
            args.standardize_data)

        existing_files = gfile.Glob(
            os.path.join(save_subdir, "{}*.pkl".format(filename)))
        filepath = os.path.join(
            save_subdir,
            "{}_{}.pkl".format(filename, 1000 + len(existing_files))[1:]
        )

        # dump the dict to a pickle file
        pickle.dump(all_results, gfile.GFile(filepath, "w"))

        # flush stfout
        sys.stdout.flush_file()


if __name__ == "__main__":

    # parse all the arguments
    args = parser.parse_args()

    # start the active learning algorithm
    main(args)
