from sklearn.model_selection import KFold
from sklearn.datasets import fetch_rcv1
import numpy as np
import itertools
import pickle
import logging

from classifiers import learn_classifiers, compute_posterior_probabilities
from cost_structure import Costs, cost_structure_1
from data_setup import get_setup_data, TRAINING_SET_END
from utils import compute_prevalence
from minecore import MineCore
from multiprocessing import Pool

logging.basicConfig(filename='cross_validation.log', format="%(asctime)s %(message)s", level=logging.DEBUG)

# Target: {(cr, cp): {(0.0, 0.1): 123, (0.0, 0.2): 321}}
def test_alpha_value(alpha_cr: float, alpha_cp: float):
    run_costs = dict()
    alpha_labels = {pair: [alpha_cr, alpha_cp] for pair in pairs}
    for train_index, test_index in k_fold.split(train_x):
        train = train_x[train_index]
        test = train_x[test_index]
        classifiers = learn_classifiers(dataset, train, labels, 10, train_index=train_index)
        posterior_probabilities = compute_posterior_probabilities(dataset, test, labels, classifiers)

        train_y = dict()
        test_y = dict()
        for i, c in enumerate(dataset.target_names):
            train_y[c] = np.asarray(dataset.target[train_index, i].todense()).squeeze()
            test_y[c] = np.asarray(dataset.target[test_index, i].todense()).squeeze()

        prior_probabilities, _ = compute_prevalence(labels, train_y)
        costs = Costs(cost_structure_1, pairs, posterior_probabilities, test_y)
        minecore = MineCore(pairs, prior_probabilities, posterior_probabilities, test_y, alpha_labels, 1.0, 1.0)
        tau_rs, tau_ps, _, cm_3 = minecore.run_plusplus(costs)
        for key, value in costs.get_third_phase_costs(cm_3, tau_rs, tau_ps)[0].items():
            prec_val = run_costs.setdefault(key, {(alpha_cr, alpha_cp): 0})
            run_costs[key][(alpha_cr, alpha_cp)] = prec_val[(alpha_cr, alpha_cp)] + value
    logging.info(f"\nRun costs for alpha {(alpha_cr, alpha_cp)}:\n{run_costs}\n")
    return run_costs


def test_alpha_values_single_pair(alpha_cr: float, alpha_cp: float, label_cr, label_cp):
    run_costs = dict()
    labels = [label_cr, label_cp]
    pairs = [(label_cr, label_cp)]
    alpha_labels = {pair: [alpha_cr, alpha_cp] for pair in pairs}
    for train_index, test_index in k_fold.split(train_x):
        train = train_x[train_index]
        test = train_x[test_index]
        classifiers = learn_classifiers(dataset, train, labels, 10, train_index=train_index)
        posterior_probabilities = compute_posterior_probabilities(dataset, test, labels, classifiers)

        train_y = dict()
        test_y = dict()

        for label in labels:
            train_y[label] = np.asarray(dataset.target[train_index, dataset.target_names.searchsorted(label)].todense()).squeeze()
            test_y[label] = np.asarray(dataset.target[test_index, dataset.target_names.searchsorted(label)].todense()).squeeze()

        prior_probabilities, _ = compute_prevalence(labels, train_y)
        costs = Costs(cost_structure_1, pairs, posterior_probabilities, test_y)
        minecore = MineCore(pairs, prior_probabilities, posterior_probabilities, test_y, alpha_labels, 1.0, 1.0)
        tau_rs, tau_ps, _, cm_3 = minecore.run_plusplus(costs)
        for key, value in costs.get_third_phase_costs(cm_3, tau_rs, tau_ps)[0].items():
            prec_val = run_costs.setdefault(key, {(alpha_cr, alpha_cp): 0})
            run_costs[key][(alpha_cr, alpha_cp)] = prec_val[(alpha_cr, alpha_cp)] + value
    logging.info(f"\nRun costs for alpha {(alpha_cr, alpha_cp)}:\n{run_costs}\n")
    return run_costs


def cross_validate_first_decimal(out_file):
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    with Pool(10) as p:
        results = p.starmap(test_alpha_value, itertools.product(alphas, alphas))

    logging.info("\n###################################\n")
    logging.info(f"Final results first decimal: {results}")

    alpha_dict = dict()
    for d in results:
        for k, v in d.items():
            val = alpha_dict.setdefault(k, {})
            for k2, v2 in v.items():
                val2 = val.setdefault(k2, 0)
                val[k2] = val2 + v2

    with open(out_file, 'wb') as f:
        pickle.dump(alpha_dict, f)


def cross_validate_second_decimal(in_file, out_file):
    with open(in_file, 'rb') as f:
        alpha_dict = pickle.load(f)

    alpha_dict = dict(map(lambda kv: (kv[0], min(kv[1], key=kv[1].get)), alpha_dict.items()))
    final_results = dict()
    with Pool(10) as p:
        for pair, alphas in alpha_dict.items():
            logging.info(f"Cross validating pair {pair}, with best alpha {alphas}")
            # alpha can be lower than 0 when its 0.1 and lower than 1 when it's 0.9 (there are no intermediates here,
            # like 0.97)
            if 0 < alphas[0] < 1:
                possible_values_cr = np.arange(alphas[0] - 0.05, alphas[0] + 0.05, step=0.01)
            elif alphas[0] == 0:
                possible_values_cr = np.arange(0, 0.05, step=0.01)
            else:
                possible_values_cr = np.arange(0.95, 1, step=0.01)

            if 0 < alphas[1] < 1:
                possible_values_cp = np.arange(alphas[1] - 0.05, alphas[1] + 0.05, step=0.01)
            elif alphas[1] == 0:
                possible_values_cp = np.arange(0, 0.05, step=0.01)
            else:
                possible_values_cp = np.arange(0.95, 1, step=0.01)

            it = itertools.product(possible_values_cr, possible_values_cp)

            results = p.starmap(
                test_alpha_values_single_pair,
                map(lambda x: itertools.chain(x, pair), it)
            )
            for d in results:
                for k, v in d.items():
                    val = final_results.setdefault(k, {})
                    for k2, v2 in v.items():
                        val2 = val.setdefault(k2, 0)
                        val[k2] = val2 + v2

    logging.info("\n############################\n")
    logging.info(f"Final results: {final_results}")
    # alpha_dict = dict()
    # for d in final_results:
    #     for k, v in d.items():
    #         val = alpha_dict.setdefault(k, {})
    #         for k2, v2 in v.items():
    #             val2 = val.setdefault(k2, 0)
    #             val[k2] = val2 + v2
    with open(out_file, 'wb') as f:
        pickle.dump(final_results, f)

    # To get the target, use this
    # alpha_dict = dict(map(lambda kv: (kv[0], min(kv[1], key=kv[1].get)), alpha_dict.items()))



# Target: {(c_r, c_p): [0.8, 0.9]} -> 0.8 and 0.9 are the k-fold optimized alpha values
if __name__ == '__main__':
    dataset = fetch_rcv1()
    train_x = dataset.data[0:TRAINING_SET_END]
    full_y_arr, quarter_y_arr, pairs, labels = get_setup_data(dataset)

    k_fold = KFold(n_splits=10)
    cross_validate_first_decimal("./pickles/alpha_dict_firstdec_3108.pkl")
    cross_validate_second_decimal("./pickles/alpha_dict_firstdec_3108.pkl", "./pickles/alpha_dict_labels_3108.pkl")