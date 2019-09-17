import pickle
import numpy as np
from sklearn.datasets import rcv1

from cost_structure import Costs, cost_structure_1
from data_setup import pairs
from emq_attempt import emq_attempt
from minecore import MineCore
from utils import compute_prevalence

with open('./pickles/al_dataset_1000', 'rb') as f:
    prob, training_sets = pickle.load(f)


dataset = rcv1.fetch_rcv1()
labels = set()
for cr, cp in pairs:
    labels.add(cr)
    labels.add(cp)

y_arr = dict()
training_y = dict()
emq_posteriors = dict()

for label in labels:
    train_idxs = list(training_sets[label])
    mask = np.ones(dataset.data.shape[0], dtype=bool)
    mask[train_idxs] = False
    y_arr[label] = np.asarray(dataset.target[mask, dataset.target_names.searchsorted(label)].todense()).squeeze()
    training_y[label] = np.asarray(dataset.target[train_idxs, dataset.target_names.searchsorted(label)].todense()).squeeze()
    prob[label] = prob[label][mask]

pos_prev, neg_prev = compute_prevalence(labels, training_y)

print("Computing EMQ")
for label in labels:
    post, pos_prior, neg_prior = emq_attempt(prob[label], pos_prev[label], neg_prev[label])
    emq_posteriors[label] = post

costs = Costs(cost_structure_1, pairs, emq_posteriors, y_arr)
minecore = MineCore(pairs, None, emq_posteriors, y_arr, None, 1.0, 1.0)

print("Running standard Minecore")
# standard_results = minecore.run(costs)

# minecore.posterior_probabilities = emq_posteriors

print("Running EMQ Minecore")
emq_results = minecore.run(costs)

print("Saving results")
# with open('./pickles/minecore_al_standard_0307.pkl', 'wb') as f:
#     pickle.dump(standard_results, f)

with open('./pickles/minecore_al_emq_0307.pkl', 'wb') as f:
    pickle.dump(emq_results, f)
