from classifiers import generate_dataset_via_active_learning, relevance_sampling_policy, learn_classifiers
from sklearn.datasets import rcv1
from data_setup import pairs
import numpy as np
import pickle


TRAIN_SIZE = 23149
TEST_SIZE = TRAIN_SIZE + 199328
INITIAL_SEED = 1000


checkpoints = {5000, 8000, 15000}
labels = set()
for cr, cp in pairs:
    labels.add(cr)
    labels.add(cp)

dataset = rcv1.fetch_rcv1()

# Initial dataset is taken at random. Number of documents is specified by INITIAL_SEED
initial_train_idx = np.random.randint(0, dataset.data.shape[0], INITIAL_SEED)
random_train = dataset.data[initial_train_idx]
classifiers = learn_classifiers(dataset, random_train, labels, 10, initial_train_idx)
test_mask = np.ones(dataset.data.shape[0], dtype=bool)

test_mask[initial_train_idx] = False
random_test = dataset.data[test_mask]

probs = dict()
training_sets = dict()

for label in labels:
    print(f"Generating dataset for label {label}")
    probs[label] = classifiers[label].predict_proba(random_test)
    probs[label], train_indexes = generate_dataset_via_active_learning(probs, relevance_sampling_policy, label, TRAIN_SIZE, dataset, checkpoints)
    training_sets[label] = train_indexes


print("Saving results")
with open(f'./pickles/al_dataset_{INITIAL_SEED}', 'wb') as f:
    pickle.dump([probs, training_sets], f)
