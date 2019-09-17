from sklearn.datasets import fetch_rcv1
import numpy as np
from classifiers import load_or_learn_classifiers, compute_posterior_probabilities, learn_classifiers
from minecore import MineCore
from utils import compute_probabilities, find_pairs
from cost_structure import Costs, cost_structure_1
import pickle

TRAINING_SET_END = 23149
SMALL_TEST_SET_END = TRAINING_SET_END + 199328
TEST_SET_START = TRAINING_SET_END
TEST_SET_END = SMALL_TEST_SET_END

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

def get_setup_data(dataset) -> ({str: np.array}, {str: np.array}, [(str, str)], {str}):
    full_test_set_end = dataset.target.shape[0]
    quarter_y_arr = dict()
    full_y_arr = dict()

    for i, c in enumerate(dataset.target_names):
        quarter_y_arr[c] = np.asarray(dataset.target[TEST_SET_START:TEST_SET_END, i].todense()).squeeze()
        full_y_arr[c] = np.asarray(dataset.target[0:TEST_SET_END, i].todense()).squeeze()


    labels = set()
    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)
    return full_y_arr, quarter_y_arr, pairs, labels


def main():
    rcv1 = fetch_rcv1()
    full_y_arr, quarter_y_arr, pairs, labels = get_setup_data(rcv1)

    classifiers = learn_classifiers(
        rcv1, rcv1.data[0:TRAINING_SET_END],
        labels,
        kfold=10,
        training_set_end=TRAINING_SET_END
    )

    posterior_probabilities = compute_posterior_probabilities(rcv1, rcv1.data[TEST_SET_START:TEST_SET_END, :], labels, classifiers)

    costs = Costs(cost_structure_1, pairs, posterior_probabilities, quarter_y_arr)
    minecore = MineCore(pairs, posterior_probabilities, quarter_y_arr)

    # tau_rs, tau_ps, cm_2, cm_3 = minecore.run(costs)
    # with open("./pickles/mle_results_220319.pkl", 'wb') as f:
    #     pickle.dump([tau_rs, tau_ps, cm_2, cm_3], f)

    trs, tps, cm2, cm3 = minecore.custom_run(costs)
    with open("./pickles/custom_results_220319_2.pkl", 'wb') as f:
        pickle.dump([trs, tps, cm2, cm3], f)


if __name__ == '__main__':
    main()
