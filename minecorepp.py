import pickle

from sklearn.datasets import fetch_rcv1
import numpy as np
from cost_structure import cost_structure_1, Costs
from data_setup import TRAINING_SET_END, get_setup_data, TEST_SET_START, TEST_SET_END
from emq_attempt import emq_attempt
from utils import compute_prevalence
from minecore import MineCore
from multiprocessing import Process

dataset = fetch_rcv1()
train_x = dataset.data[0:TRAINING_SET_END]
train_y = dict()
full_y_arr, quarter_y_arr, pairs, labels = get_setup_data(dataset)

for i, c in enumerate(dataset.target_names):
    train_y[c] = np.asarray(dataset.target[0:TRAINING_SET_END, i].todense()).squeeze()


with open('./pickles/post_prob.pkl', 'rb') as f:
    posterior_probabilities = pickle.load(f)

with open('./pickles/alpha_dict_labels_3108.pkl', 'rb') as f:
    alphas = pickle.load(f)

alphas = dict(map(lambda kv: (kv[0], min(kv[1], key=kv[1].get)), alphas.items()))
# classifiers = learn_classifiers(dataset, train_x, labels, 10, training_set_end=TRAINING_SET_END)
# posterior_probabilities = compute_posterior_probabilities(dataset, dataset.data[TEST_SET_START:TEST_SET_END, :], labels, classifiers)
prior_probabilities, neg_priors = compute_prevalence(labels, train_y)
# new_priors = dict()
# for label in labels:
#     print(f"Updating probabilities for label: {label}")
#     _, em_pos, em_neg = emq_attempt(posterior_probabilities[label], prior_probabilities[label], neg_priors[label])
#     new_priors[label] = em_pos

# alphas = dict(map(lambda kv: (kv[0], (1.0, 1.0)), alphas.items()))
ro_r = 0.50
ro_p = 0.99
costs = Costs(cost_structure_1, pairs, posterior_probabilities, quarter_y_arr, alphas=alphas,
              prior_probabilities=prior_probabilities, ro_r=ro_r, ro_p=ro_p)
minecore = MineCore(pairs, prior_probabilities, posterior_probabilities, quarter_y_arr, alphas, ro_r, ro_p)


def run_and_save(run_func, costs, file_name):
    tau_rs, tau_ps, cm_2, cm_3 = run_func(costs)
    with open(file_name, 'wb') as f:
        pickle.dump([tau_rs, tau_ps, cm_2, cm_3], f)

p1 = Process(target=run_and_save, args=(minecore.run_plusplus, costs, '/home/alessio/minecore/pickles/minecorepp_1309_rorp_050_99_alphas.pkl'))
p2 = Process(target=run_and_save, args=(minecore.run, costs, '/home/alessio/minecore/pickles/minecore_1309_rorp_050_99_alphas.pkl'))
p1.start()
p2.start()
p1.join()
p2.join()
