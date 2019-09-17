import numpy as np
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


def load_or_learn_classifiers(dataset, train_x, labels, kfold: int, training_set_end: int) -> {str: CalibratedClassifierCV}:
    classifiers = load_classifiers()
    if classifiers is None:
        classifiers = learn_classifiers(dataset, train_x, labels, kfold, training_set_end)
        save_classifiers(classifiers)
    return classifiers


def learn_classifiers(dataset, train_x, labels, kfold: int, train_index=None, training_set_end=None) -> {str: CalibratedClassifierCV}:
    classifiers = dict()
    for label in labels:
        label_idx = dataset.target_names.tolist().index(label)
        learner = LinearSVC(loss='hinge')
        if train_index is not None:
            target = dataset.target[train_index]
            train_y = np.asarray(target[:, [label_idx]].todense()).squeeze()
        else:
            train_y = np.asarray(dataset.target[0:training_set_end, [label_idx]].todense()).squeeze()

        print(label, label_idx, train_y.sum())
        label_kfold = min(train_y.sum(), kfold)
        calibrator = CalibratedClassifierCV(learner, cv=label_kfold, method='sigmoid')
        classifiers[label] = calibrator.fit(train_x, train_y)
    return classifiers


def compute_posterior_probabilities(dataset, test_x, labels, classifiers: {str: CalibratedClassifierCV}) -> np.array:
    posterior_probabilities = dict()
    for label in labels:
        label_idx = dataset.target_names.tolist().index(label)
        print(label, label_idx)
        clf = classifiers[label]
        proba = clf.predict_proba(test_x)
        posterior_probabilities[label] = proba
    return posterior_probabilities


def save_classifiers(classifiers: {str: CalibratedClassifierCV}):
    with open('./pickles/classifiers_TOIS.pkl', mode='wb') as outputfile:
        pickle.dump(classifiers, outputfile)


def load_classifiers() -> {str: CalibratedClassifierCV}:
    try:
        with open('./pickles/classifiers_TOIS.pkl', mode='rb') as inputfile:
            classifiers = pickle.load(inputfile)
        return classifiers
    except FileNotFoundError:
        return None


def generate_dataset_via_active_learning(probs, policy, label, count, dataset, checkpoints, kfold=10, batch_size=1000):
    train_idxs = set()
    prob = probs[label][:, 1]
    while len(train_idxs) < count:
        to_add = min(batch_size, count - len(train_idxs))

        rank_idxs = policy(prob)
        rank_idxs = [idx for idx in rank_idxs if idx not in train_idxs]
        rank_idxs = rank_idxs[:to_add]
        train_idxs.update(rank_idxs)

        # all_train_idxs = sorted(list(range(0, count)) + list(train_idxs))

        train_idxs_list = list(train_idxs)
        # retrain
        train_X = dataset.data[train_idxs_list]
        label_idx = dataset.target_names.tolist().index(label)
        learner = LinearSVC(loss='hinge')
        train_y = np.asarray(dataset.target[train_idxs_list, label_idx].todense()).squeeze()
        print(label, len(train_idxs), train_y.sum())
        label_kfold = min(train_y.sum(), kfold)
        calibrator = CalibratedClassifierCV(learner, cv=label_kfold, method='sigmoid')
        clf = calibrator.fit(train_X, train_y)

        # update of posterior probabilities
        prob = clf.predict_proba(dataset.data)[:, 1]
        if len(train_idxs) in checkpoints:
            save_al_dataset(prob, train_idxs)

    train_idxs_bool = np.zeros_like(prob)
    train_idxs_bool[train_idxs_list] = 1

    y_r = np.asarray(dataset.target[:, dataset.target_names.searchsorted(label)].todense()).squeeze()


    prob[np.logical_and(train_idxs_bool == 1, y_r == 1)] = 1
    prob[np.logical_and(train_idxs_bool == 1, y_r == 0)] = 0

    return np.array([1-prob, prob]).T, train_idxs


def uncertainty_sampling_policy(prob):
        rank = np.abs(prob - 0.5)
        return np.argsort(rank)

def relevance_sampling_policy(prob):
    return np.argsort(1 - prob)


def save_al_dataset(prob, train_idxs):
    with open(f'./pickles/al_dataset_idxs_{len(train_idxs)}.pkl', 'wb') as f:
        pickle.dump([prob, train_idxs], f)
