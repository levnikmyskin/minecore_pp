import numpy as np

from postemq_analysis import get_emq_better_posteriors
from utils import compute_prevalence


def compute_prior_probabilites(labels, _posterior_probs, column):
    prior_probs = dict()
    for label in labels:
        prior_probs[label] = _posterior_probs[label][:, column].mean()
    return prior_probs


def should_stop(pos_prior_probs, current_pos_prior_probs, neg_prior_probs, current_neg_prior_probs, epsilon, s):
    val = abs(current_pos_prior_probs - pos_prior_probs) + abs(current_neg_prior_probs - neg_prior_probs)
    print(f"Difference between priors: {val}, s: {s}")
    # if s < 10:
    #     return False
    return val < epsilon


def emq_attempt(_posterior_probs, pos_prior_probs, neg_prior_probs, epsilon=1e-10):
    # pos_prior_probs and neg_prior_probs will keep the prior probs at s - 1 step
    s = 0
    stopping_condition = False
    pos_prior_zero = pos_prior_probs
    neg_prior_zero = neg_prior_probs
    posterior_zero = _posterior_probs.copy()
    current_neg_prior_probs = neg_prior_probs
    current_pos_prior_probs = pos_prior_probs
    while not stopping_condition:
        s += 1

        neg_numerator = posterior_zero[:, 0] * (current_neg_prior_probs / neg_prior_zero)
        pos_numerator = posterior_zero[:, 1] * (current_pos_prior_probs / pos_prior_zero)

        denominator = pos_numerator + neg_numerator

        # TODO test all of these sum to 1
        current_posteriors = np.array([neg_numerator / denominator, pos_numerator / denominator]).T
        current_neg_prior_probs = current_posteriors[:, 0].mean()
        current_pos_prior_probs = current_posteriors[:, 1].mean()

        stopping_condition = should_stop(pos_prior_probs, current_pos_prior_probs, neg_prior_probs, current_neg_prior_probs, epsilon, s)
        pos_prior_probs = current_pos_prior_probs
        neg_prior_probs = current_neg_prior_probs
        _posterior_probs = current_posteriors
    return _posterior_probs, pos_prior_probs, neg_prior_probs


# def emq_new_attempt(_posterior_probs, prior_probs, labels, epsilon=1e-10):
#     s = 0
#     stopping_condition = False
#     pos_prior_zero = prior_probs
#     posterior_zero = _posterior_probs.copy()
#     current_pos_prior_probs = prior_probs
#     while not stopping_condition:
#         s += 1
#         numerator = posterior_zero[:, 1] * (current_pos_prior_probs / pos_prior_zero)


if __name__ == '__main__':
    from cost_structure import Costs, cost_structure_1
    from sklearn.datasets import fetch_rcv1
    from minecore import MineCore
    import pickle

    rcv1 = fetch_rcv1()
    TRAINING_SET_END = 23149
    SMALL_TEST_SET_END = TRAINING_SET_END + 199328
    TEST_SET_START = TRAINING_SET_END
    FULL_TEST_SET_END = rcv1.target.shape[0]
    TEST_SET_END = SMALL_TEST_SET_END

    quarter_y_arr = dict()
    full_y_arr = dict()
    training_y_arr = dict()

    for i, c in enumerate(rcv1.target_names):
        quarter_y_arr[c] = np.asarray(rcv1.target[TEST_SET_START:TEST_SET_END, i].todense()).squeeze()
        full_y_arr[c] = np.asarray(rcv1.target[0:TEST_SET_END, i].todense()).squeeze()
        training_y_arr[c] = np.asarray(rcv1.target[0:TRAINING_SET_END, i].todense()).squeeze()

    with open('./pickles/post_prob.pkl', 'rb') as f:
        posterior_probs = pickle.load(f)

    pairs = [('M12', 'M14'), ('M12', 'CCAT'), ('M12', 'M132'), ('M12', 'E21'), ('M12', 'M131'), ('M132', 'GPOL'),
             ('M132', 'CCAT'), ('M132', 'M12'), ('M132', 'M131'), ('M132', 'GCAT'), ('M131', 'CCAT'), ('M131', 'M132'),
             ('M131', 'E12'), ('M131', 'ECAT'), ('M131', 'M12'), ('E12', 'M11'), ('E12', 'GDIP'), ('E12', 'E212'),
             ('E12', 'M131'), ('E12', 'E21'), ('C21', 'C17'), ('C21', 'C15'), ('C21', 'ECAT'), ('C21', 'C31'),
             ('C21', 'M141'), ('E212', 'GPOL'), ('E212', 'E12'), ('E212', 'M12'), ('E212', 'MCAT'), ('E212', 'C17'),
             ('GCRIM', 'E212'), ('GCRIM', 'C15'), ('GCRIM', 'C18'), ('GCRIM', 'GDIP'), ('GCRIM', 'GPOL'), ('C24', 'GDIP'),
             ('C24', 'C15'), ('C24', 'C31'), ('C24', 'MCAT'), ('C24', 'C21'), ('GVIO', 'C21'), ('GVIO', 'C24'), ('GVIO', 'CCAT'),
             ('GVIO', 'ECAT'), ('GVIO', 'GCRIM'), ('C13', 'M12'), ('C13', 'C15'), ('C13', 'GPOL'), ('C13', 'M14'), ('C13', 'MCAT'),
             ('GDIP', 'C31'), ('GDIP', 'E12'), ('GDIP', 'CCAT'), ('GDIP', 'ECAT'), ('GDIP', 'GPOL'), ('C31', 'C151'),
             ('C31', 'C15'), ('C31', 'ECAT'), ('C31', 'C21'), ('C31', 'M14'), ('C181', 'C151'), ('C181', 'GCAT'),
             ('C181', 'C152'), ('C181', 'C15'), ('C181', 'C17'), ('M141', 'ECAT'), ('M141', 'GCAT'), ('M141', 'C24'),
             ('M141', 'C31'), ('M141', 'C21'), ('M11', 'ECAT'), ('M11', 'C152'), ('M11', 'M132'), ('M11', 'M13'),
             ('M11', 'CCAT'), ('E21', 'C31'), ('E21', 'M12'), ('E21', 'MCAT'), ('E21', 'E12'), ('E21', 'GPOL'),
             ('C17', 'MCAT'), ('C17', 'C152'), ('C17', 'C15'), ('C17', 'C18'), ('C17', 'ECAT'), ('M13', 'E21'),
             ('M13', 'M11'), ('M13', 'GCAT'), ('M13', 'E12'), ('M13', 'ECAT'), ('C18', 'E12'), ('C18', 'GCAT'),
             ('C18', 'C152'), ('C18', 'C15'), ('C18', 'C17'), ('GPOL', 'MCAT'), ('GPOL', 'CCAT'), ('GPOL', 'GCRIM'),
             ('GPOL', 'E21'), ('GPOL', 'GVIO'), ('C152', 'M11'), ('C152', 'C17'), ('C152', 'C31'), ('C152', 'C181'),
             ('C152', 'C18'), ('M14', 'M132'), ('M14', 'M13'), ('M14', 'GCAT'), ('M14', 'C24'), ('M14', 'C31'), ('C151', 'C181'),
             ('C151', 'C18'), ('C151', 'C17'), ('C151', 'C31'), ('C151', 'C152'), ('ECAT', 'GVIO'), ('ECAT', 'C17'),
             ('ECAT', 'M13'), ('ECAT', 'GPOL'), ('ECAT', 'MCAT')]

    labels = set()
    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)

    pos_prevalences, neg_prevalences = compute_prevalence(labels, training_y_arr)

    def save(mc, costs, name):
        tau_rs, tau_ps, cm_2, cm_3 = mc.run(costs)
        with open(name, 'wb') as f:
            pickle.dump([tau_rs, tau_ps, cm_2, cm_3], f)

    # Before EMQ
    # costs = Costs(cost_structure_1, pairs, posterior_probs, quarter_y_arr)
    # mc = MineCore(pairs, posterior_probs, quarter_y_arr)
    # t1 = threading.Thread(target=save, args=(mc, costs, "before_emq.pkl"))
    # t1.start()

    # After EMQ
    new_posteriors, pos_priors = emq_new_attempt(posterior_probs, pos_prevalences, labels)

    with open('./pickles/newemq_posteriors_0607.pkl', 'wb') as f:
        pickle.dump(new_posteriors, f)

    # emq_better_posteriors = get_emq_better_posteriors(labels, quarter_y_arr, posterior_probs, new_posteriors, lambda m: m['TNM'] + m['TPM'])
    costs = Costs(cost_structure_1, pairs, new_posteriors, quarter_y_arr)
    mc = MineCore(pairs, None, new_posteriors, quarter_y_arr, None, 1, 1)
    #
    # # t1.join()
    save(mc, costs, "after_newemq_0607.pkl")
