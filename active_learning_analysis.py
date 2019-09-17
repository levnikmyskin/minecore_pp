import pickle
import numpy as np
from sklearn.datasets import rcv1
from data_setup import pairs
from emq_attempt import emq_attempt, emq_new_attempt
from postemq_analysis import get_contingency_matrix, accuracy
from plot_utils import show_distribution_drift_graph
import csv

from similarity_check import check_similarity
from utils import compute_prevalence


def prevalence_analysis():
    csv_file = open('al_results.tsv', 'w')
    writer = csv.writer(csv_file, delimiter='\t', quotechar='|')
    writer.writerow(['Label', 'Train prev', 'Test prev', 'n_positive_train', 'n_positive_test', 'train_dataset_length', 'test_dataset_length'])


    train_prev = list()
    test_prev = list()

    for label in labels:
        train_idxs = list(training_sets[label])
        mask = np.ones(dataset.data.shape[0], dtype=bool)
        mask[train_idxs] = False
        train_y = np.asarray(dataset.target[train_idxs, dataset.target_names.searchsorted(label)].todense()).squeeze()
        test_y = np.asarray(dataset.target[mask, dataset.target_names.searchsorted(label)].todense()).squeeze()
        train_prev.append(train_y.mean())
        test_prev.append(test_y.mean())
        writer.writerow([label, train_y.mean(), test_y.mean(), train_y.sum(), test_y.sum(), train_y.shape[0], test_y.shape[0]])

    print(f"Train prev mean: {np.array(train_prev).mean()}\nTest prev mean: {np.array(test_prev).mean()}")
    csv_file.close()


if __name__ == '__main__':
    with open('./pickles/al_dataset_1000', 'rb') as f:
        prob, training_sets = pickle.load(f)

    dataset = rcv1.fetch_rcv1()
    labels = set()
    cr_set = set()
    cp_set = set()
    training_y = dict()
    y_arr = dict()
    emq_post = dict()
    emq_pos_priors = dict()
    emq_neg_priors = dict()

    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)
        cr_set.add(cr)
        cp_set.add(cp)

    for label in labels:
        train_idxs = list(training_sets[label])
        mask = np.ones(dataset.data.shape[0], dtype=bool)
        mask[train_idxs] = False
        y_arr[label] = np.asarray(dataset.target[mask, dataset.target_names.searchsorted(label)].todense()).squeeze()
        training_y[label] = np.asarray(dataset.target[train_idxs, dataset.target_names.searchsorted(label)].todense()).squeeze()
        prob[label] = prob[label][mask]


    pos_prev, neg_prev = compute_prevalence(labels, training_y)
    true_pos_prev, true_neg_prev = compute_prevalence(labels, y_arr)

    show_distribution_drift_graph(pos_prev, true_pos_prev, labels)


    for label in labels:
        print(f"Updating probabilities for label: {label}")
        emq_post[label], emq_pos_priors[label], emq_neg_priors[label] = emq_attempt(prob[label], pos_prev[label], neg_prev[label])

    classifier_matrix, emq_matrix = get_contingency_matrix(labels, prob, emq_post, y_arr)
    labels_len = len(labels)
    for key, val in classifier_matrix.items():
        classifier_matrix[key] = val / labels_len
        emq_matrix[key] = emq_matrix[key] / labels_len

    print(f"Classifier matrix: {classifier_matrix}\nEMQ Matrix: {emq_matrix}")

    mle_accuracy = accuracy(classifier_matrix)
    emq_accuracy = accuracy(emq_matrix)

    print(f"MLE accuracy: {mle_accuracy}\nEMQ accuracy: {emq_accuracy}")

    train_errs, emq_errs = check_similarity(pos_prev, neg_prev, emq_pos_priors, emq_neg_priors, true_pos_prev, true_neg_prev, labels)

    print(f"Train errs: {train_errs}\nEMQ errs: {emq_errs}")

