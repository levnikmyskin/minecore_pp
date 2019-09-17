import numpy as np
from emq_attempt import should_stop
from utils import compute_prevalence


def absolute_error(extim_pos, extim_neg, true_pos, true_neg):
    return (abs(extim_pos - true_pos) + abs(extim_neg - true_neg)) / 2


def check_similarity(pos_training_priors, neg_training_priors, emq_pos_priors, emq_neg_priors, pos_test_priors, neg_test_priors, labels):
    training_abs_errors = 0
    emq_abs_errors = 0
    for label in labels:
        training_abs_errors += absolute_error(pos_training_priors[label], neg_training_priors[label], pos_test_priors[label], neg_test_priors[label])
        emq_abs_errors += absolute_error(emq_pos_priors[label], emq_neg_priors[label], pos_test_priors[label], neg_test_priors[label])
    return training_abs_errors, emq_abs_errors


def emq_attempt(_posterior_probs, pos_prior_probs, neg_prior_probs, epsilon=1e-10):
    # pos_prior_probs and neg_prior_probs will keep the prior probs at s - 1 step
    s = 0
    stopping_condition = False
    while not stopping_condition:
        s += 1
        current_neg_prior_probs = _posterior_probs[:, 0].mean()
        current_pos_prior_probs = _posterior_probs[:, 1].mean()

        neg_numerator = _posterior_probs[:, 0] * (current_neg_prior_probs / neg_prior_probs)
        pos_numerator = _posterior_probs[:, 1] * (current_pos_prior_probs / pos_prior_probs)

        neg_denominator = _posterior_probs[:, 0] * (current_neg_prior_probs / neg_prior_probs)
        pos_denominator = _posterior_probs[:, 1] * (current_pos_prior_probs / pos_prior_probs)
        denominator = neg_denominator + pos_denominator

        current_posteriors = np.array([neg_numerator / denominator, pos_numerator / denominator]).T
        stopping_condition = should_stop(pos_prior_probs, current_pos_prior_probs, neg_prior_probs, current_neg_prior_probs, epsilon, s)

        pos_prior_probs = current_pos_prior_probs
        neg_prior_probs = current_neg_prior_probs
        _posterior_probs = current_posteriors
    return _posterior_probs, pos_prior_probs, neg_prior_probs, s


if __name__ == '__main__':
    from sklearn.datasets import fetch_rcv1
    import pickle

    rcv1 = fetch_rcv1()
    TRAINING_SET_END = 23149
    SMALL_TEST_SET_END = TRAINING_SET_END + 199328
    TEST_SET_START = TRAINING_SET_END
    FULL_TEST_SET_END = rcv1.target.shape[0]
    TEST_SET_END = SMALL_TEST_SET_END

    quarter_y_arr = dict()
    full_y_arr = dict()
    quarter_x_arr = dict()

    for i, c in enumerate(rcv1.target_names):
        quarter_y_arr[c] = np.asarray(rcv1.target[TEST_SET_START:TEST_SET_END, i].todense()).squeeze()
        full_y_arr[c] = np.asarray(rcv1.target[0:TEST_SET_END, i].todense()).squeeze()
        quarter_x_arr[c] = np.asarray(rcv1.target[0:TRAINING_SET_END, i].todense()).squeeze()

    with open('./pickles/post_prob.pkl', 'rb') as f:
        posterior_probs = pickle.load(f)
    pairs = [('M12', 'M14'), ('M12', 'CCAT'), ('M12', 'M132'), ('M12', 'E21'), ('M12', 'M131'), ('M132', 'GPOL'),
             ('M132', 'CCAT'), ('M132', 'M12'), ('M132', 'M131'), ('M132', 'GCAT'), ('M131', 'CCAT'), ('M131', 'M132'),
             ('M131', 'E12'), ('M131', 'ECAT'), ('M131', 'M12'), ('E12', 'M11'), ('E12', 'GDIP'), ('E12', 'E212'),
             ('E12', 'M131'), ('E12', 'E21'), ('C21', 'C17'), ('C21', 'C15'), ('C21', 'ECAT'), ('C21', 'C31'),
             ('C21', 'M141'), ('E212', 'GPOL'), ('E212', 'E12'), ('E212', 'M12'), ('E212', 'MCAT'), ('E212', 'C17'),
             ('GCRIM', 'E212'), ('GCRIM', 'C15'), ('GCRIM', 'C18'), ('GCRIM', 'GDIP'), ('GCRIM', 'GPOL'),
             ('C24', 'GDIP'),
             ('C24', 'C15'), ('C24', 'C31'), ('C24', 'MCAT'), ('C24', 'C21'), ('GVIO', 'C21'), ('GVIO', 'C24'),
             ('GVIO', 'CCAT'),
             ('GVIO', 'ECAT'), ('GVIO', 'GCRIM'), ('C13', 'M12'), ('C13', 'C15'), ('C13', 'GPOL'), ('C13', 'M14'),
             ('C13', 'MCAT'),
             ('GDIP', 'C31'), ('GDIP', 'E12'), ('GDIP', 'CCAT'), ('GDIP', 'ECAT'), ('GDIP', 'GPOL'), ('C31', 'C151'),
             ('C31', 'C15'), ('C31', 'ECAT'), ('C31', 'C21'), ('C31', 'M14'), ('C181', 'C151'), ('C181', 'GCAT'),
             ('C181', 'C152'), ('C181', 'C15'), ('C181', 'C17'), ('M141', 'ECAT'), ('M141', 'GCAT'), ('M141', 'C24'),
             ('M141', 'C31'), ('M141', 'C21'), ('M11', 'ECAT'), ('M11', 'C152'), ('M11', 'M132'), ('M11', 'M13'),
             ('M11', 'CCAT'), ('E21', 'C31'), ('E21', 'M12'), ('E21', 'MCAT'), ('E21', 'E12'), ('E21', 'GPOL'),
             ('C17', 'MCAT'), ('C17', 'C152'), ('C17', 'C15'), ('C17', 'C18'), ('C17', 'ECAT'), ('M13', 'E21'),
             ('M13', 'M11'), ('M13', 'GCAT'), ('M13', 'E12'), ('M13', 'ECAT'), ('C18', 'E12'), ('C18', 'GCAT'),
             ('C18', 'C152'), ('C18', 'C15'), ('C18', 'C17'), ('GPOL', 'MCAT'), ('GPOL', 'CCAT'), ('GPOL', 'GCRIM'),
             ('GPOL', 'E21'), ('GPOL', 'GVIO'), ('C152', 'M11'), ('C152', 'C17'), ('C152', 'C31'), ('C152', 'C181'),
             ('C152', 'C18'), ('M14', 'M132'), ('M14', 'M13'), ('M14', 'GCAT'), ('M14', 'C24'), ('M14', 'C31'),
             ('C151', 'C181'),
             ('C151', 'C18'), ('C151', 'C17'), ('C151', 'C31'), ('C151', 'C152'), ('ECAT', 'GVIO'), ('ECAT', 'C17'),
             ('ECAT', 'M13'), ('ECAT', 'GPOL'), ('ECAT', 'MCAT')]

    labels = set()
    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)

    train_pos_prevalences, train_neg_prevalences = compute_prevalence(labels, quarter_x_arr)
    true_pos_prevalences, true_neg_prevalences = compute_prevalence(labels, quarter_y_arr)
    new_posteriors = dict()
    pos_priors = dict()
    neg_priors = dict()
    for label in labels:
        print(f"Updating probabilities for label: {label}")
        new_posteriors[label], pos_priors[label], neg_priors[label], s = emq_attempt(posterior_probs[label], train_pos_prevalences[label], train_neg_prevalences[label])

    train_errs, emq_errs = check_similarity(
        train_pos_prevalences,
        train_neg_prevalences,
        pos_priors,
        neg_priors,
        true_pos_prevalences,
        true_neg_prevalences,
        labels
    )
    print("----------------------\n")
    print(f"Absolute errors:\ntraining: {train_errs}\nEMQ: {emq_errs}")
