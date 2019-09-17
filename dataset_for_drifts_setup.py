import random
import threading
from multiprocessing.pool import Pool

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_rcv1
import numpy as np
from sklearn.svm import LinearSVC

from classifiers import load_or_learn_classifiers, compute_posterior_probabilities
from emq_attempt import emq_attempt
from minecore import MineCore
from similarity_check import check_similarity
from utils import compute_probabilities, find_pairs, compute_prevalence
from cost_structure import Costs, cost_structure_1
from data_setup import get_setup_data, TRAINING_SET_END, TEST_SET_START, TEST_SET_END
import pickle


def learn_classifiers(train_x, quarter_y_arr, labels, kfold: int) -> {str: CalibratedClassifierCV}:
    classifiers = dict()
    for label in labels:
        learner = LinearSVC(loss='hinge')
        train_y = quarter_y_arr[label]
        label_kfold = min(train_y.sum(), kfold)
        calibrator = CalibratedClassifierCV(learner, cv=label_kfold, method='sigmoid')
        classifiers[label] = calibrator.fit(train_x, train_y)
    return classifiers


def __shrink(train_x, train_y, number_to_shrink: int):
    negative_indexes = np.where(train_y == 0)[0]
    to_delete = np.random.choice(negative_indexes, number_to_shrink, replace=False)  # Get unique indexes
    mask = np.ones(len(train_y), dtype=bool)
    mask[to_delete] = False
    new_x = train_x[mask]
    new_y = train_y[mask]
    return new_x, new_y


def shrink_dataset_for_drift(test_number: int):
    print(f"Kicking in with run number {test_number}")
    dataset = fetch_rcv1()
    full_y_arr, _, pairs, labels = get_setup_data(dataset)
    shrink_20 = int((TRAINING_SET_END * 20) / 100)
    new_x = dataset.data[0:TRAINING_SET_END]
    new_y = np.asarray(dataset.target[0:TRAINING_SET_END].todense()).squeeze()
    # Shrink up to 80%
    for i in range(4):
        print(f"Running shrinking phase number {i}")
        training_y_arr = dict()
        quarter_y_arr = dict()
        new_x, new_y = __shrink(new_x, new_y, shrink_20)
        training_set_end = new_x.shape[0]
        for j, c in enumerate(dataset.target_names):
            quarter_y_arr[c] = np.asarray(dataset.target[TEST_SET_START:TEST_SET_END, j].todense()).squeeze()
            training_y_arr[c] = new_y[0:training_set_end, j]

        print(f"Computing those amazing classifiers")
        classifiers = learn_classifiers(new_x, training_y_arr, labels, kfold=10)
        posterior_probabilities = compute_posterior_probabilities(dataset, dataset.data[TEST_SET_START:TEST_SET_END, :], labels, classifiers)

        costs = Costs(cost_structure_1, pairs, posterior_probabilities, quarter_y_arr)
        minecore = MineCore(pairs, posterior_probabilities, quarter_y_arr)
        pos_prevalences, neg_prevalences = compute_prevalence(labels, training_y_arr)

        with open(f'pickles/shrink_tests/posterior_prob_random_{test_number}_{i}', 'wb') as f:
            pickle.dump(posterior_probabilities, f)

        print(f"Posterior probabilities MLE saved")

        def save(mc, costs, name):
            tau_rs, tau_ps, cm_2, cm_3 = mc.run_plusplus(costs)
            with open(name, 'wb') as f:
                pickle.dump([tau_rs, tau_ps, cm_2, cm_3], f)

        print(f"Computing EMQ posteriors at incredibly high speed")
        new_posteriors = dict()
        for label in labels:
            new_posteriors[label], s = emq_attempt(posterior_probabilities[label], pos_prevalences[label], neg_prevalences[label])

        with open(f'pickles/shrink_tests/emq_posteriors_random_{test_number}_{i}', 'wb') as f:
            pickle.dump(new_posteriors, f)

        t1 = threading.Thread(target=save, args=(minecore, costs, f"pickles/shrink_tests/mle_results_random_{test_number}_{i}"))
        t1.start()
        print(f"Thread with MLE started, get ready to have your pc on fire")

        emq_mc = MineCore(pairs, new_posteriors, quarter_y_arr)
        emq_costs = Costs(cost_structure_1, pairs, new_posteriors, quarter_y_arr)
        print(f"Starting Minecore-EMQ, call the firemen")
        save(emq_mc, emq_costs, f"pickles/shrink_tests/emq_results_random_{test_number}_{i}")

        t1.join()


def shrink_dataset_to_80():
    training_y_arr = dict()
    quarter_y_arr = dict()
    dataset = fetch_rcv1()
    full_y_arr, _, pairs, labels = get_setup_data(dataset)
    shrink_80 = int((TRAINING_SET_END * 80) / 100)

    new_x = dataset.data[0:TRAINING_SET_END]
    new_y = np.asarray(dataset.target[0:TRAINING_SET_END].todense()).squeeze()
    new_x, new_y = __shrink(new_x, new_y, shrink_80)
    training_set_end = new_x.shape[0]

    for j, c in enumerate(dataset.target_names):
        quarter_y_arr[c] = np.asarray(dataset.target[TEST_SET_START:TEST_SET_END, j].todense()).squeeze()
        training_y_arr[c] = new_y[0:training_set_end, j]

    print(f"Computing those amazing classifiers")
    classifiers = learn_classifiers(new_x, training_y_arr, labels, kfold=10)
    posterior_probabilities = compute_posterior_probabilities(dataset, dataset.data[TEST_SET_START:TEST_SET_END, :], labels, classifiers)
    pos_prevalences, neg_prevalences = compute_prevalence(labels, training_y_arr)
    true_pos_prevalences, true_neg_prevalences = compute_prevalence(labels, quarter_y_arr)

    new_posteriors = dict()
    pos_priors = dict()
    neg_priors = dict()
    for label in labels:
        new_posteriors[label], pos_priors[label], neg_priors[label] = emq_attempt(posterior_probabilities[label], pos_prevalences[label], neg_prevalences[label])

    train_errs, emq_errs = check_similarity(
        pos_prevalences,
        neg_prevalences,
        pos_priors,
        neg_priors,
        true_pos_prevalences,
        true_neg_prevalences,
        labels
    )
    print("----------------------\n")
    print(f"Absolute errors:\ntraining: {train_errs}\nEMQ: {emq_errs}")



if __name__ == '__main__':
    shrink_dataset_to_80()


