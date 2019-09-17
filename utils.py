import pickle
import numpy as np


def load_posterior_probabilities():
    with open('./pickles/posterior_probabilities_TOIS.pkl', mode='rb') as inputfile:
        posterior_probabilities = pickle.load(inputfile)
    return posterior_probabilities


def save_posterior_probabilities(posterior_probabilities):
    with open('./pickles/posterior_probabilities_TOIS.pkl', mode='wb') as outputfile:
        pickle.dump(posterior_probabilities, outputfile)


def compute_probabilities(rcv1, ys):
    cr_values = dict()
    cp_cr_values = dict()
    for cr in rcv1.target_names:
        labels_cr = ys[cr]
        cr_count = labels_cr.sum()
        p_cr = cr_count/labels_cr.shape[0]
        cr_values[cr] = p_cr
        # Since we only take cp_cr pairs with prob. 0.01 <= 0.2 there's no
        # need to skip the current cr class here (cr, cp prob will be 1)
        for cp in rcv1.target_names:
            labels_cp = ys[cp]
            cp_cr_count = labels_cp[np.logical_and(labels_cr == 1, labels_cp == 1)].sum()
            if cr_count != 0:
                p_cp_cr = cp_cr_count/cr_count
            else:
                p_cp_cr = 0
            cp_cr_values[(cp, cr)] = p_cp_cr
    return cr_values, cp_cr_values


def find_pairs(dataset, cr_bins, cp_cr_bins, cr_values, cp_cr_values):
    pairs = list()
    cr_min, cr_max = cr_bins
    cp_cr_min, cp_cr_max = cp_cr_bins
    for cr in dataset.target_names:
        p_cr = cr_values[cr]
        if cr_min <= p_cr <= cr_max:
            for cp in dataset.target_names:
                p_cp_cr = cp_cr_values[(cp, cr)]
                if cp_cr_min <= p_cp_cr <= cp_cr_max:
                    pairs.append((cr, cp))
    return pairs


def evaluate_model(dataset, labels, posterior_probabilities):
    gtp = 0
    gtn = 0
    gfp = 0
    gfn = 0
    f1_values = list()
    accu_values = list()
    for label in labels:
        label_idx = dataset.target_names.tolist().index(label)
        evaluation = ((np.asarray(dataset.target[TRAINING_SET_END:TEST_SET_END, [label_idx]].todense()).squeeze() * 2) + (
                posterior_probabilities[label][:, 1] > 0.5))
        tp = (evaluation == 3).sum()
        tn = (evaluation == 0).sum()
        fp = (evaluation == 1).sum()
        fn = (evaluation == 2).sum()
        gtp += tp
        gtn += tn
        gfp += fp
        gfn += fn
        accu_values.append((tp + tn) / (tp + tn + fp + fn))
        f1_values.append((2 * tp) / (2 * tp + fp + fn))
        print(label, tp, tn, fp, fn, (tp + tn) / (tp + tn + fp + fn), (2 * tp) / (2 * tp + fp + fn))

    print('macro', np.mean(accu_values), np.mean(f1_values))
    print("micro", gtp, gtn, gfp, gfn, (gtp + gtn) / (gtp + gtn + gfp + gfn), (2 * gtp) / (2 * gtp + gfp + gfn))


def compute_prevalence(labels, dataset):
    pos_prevalences = dict()
    neg_prevalences = dict()
    for label in labels:
        label_target = dataset[label]
        prob_pos = label_target.mean()
        pos_prevalences[label] = prob_pos
        neg_prevalences[label] = 1 - prob_pos
    return pos_prevalences, neg_prevalences

def compute_document_prevalence(labels, dataset):
    pos_prevalences = dict()
    neg_prevalences = dict()
    for label in labels:
        pass


