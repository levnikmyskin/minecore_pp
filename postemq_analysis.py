import numpy as np
from pprint import pprint


def get_contingency_matrix_one_label(truth_posteriors, extim_posteriors):
    return {
        'TPM': np.logical_and(truth_posteriors == 1, extim_posteriors[:, 1] > 0.5).sum(),
        'FPM': np.logical_and(truth_posteriors == 0, extim_posteriors[:, 1] > 0.5).sum(),
        'TNM': np.logical_and(truth_posteriors == 0, extim_posteriors[:, 0] > 0.5).sum(),
        'FNM': np.logical_and(truth_posteriors == 1, extim_posteriors[:, 0] > 0.5).sum()
    }


def precision(matrix, true_value, false_value):
    if matrix[true_value] == 0:
        return 0
    return matrix[true_value] / (matrix[true_value] + matrix[false_value])


def accuracy(matrix):
    return (matrix['TPM'] + matrix['TNM']) / sum(matrix.values())


def has_no_positives(cont):
    return cont['TPM'] == 0 and cont['FPM'] == 0


def get_emq_better_posteriors(labels, truth_posteriors, mle_posteriors, emq_posteriors, compare_value_func) -> {str}:
    emq_better = set()
    for label in labels:
        mle_matrix = get_contingency_matrix_one_label(truth_posteriors[label], mle_posteriors[label])
        emq_matrix = get_contingency_matrix_one_label(truth_posteriors[label], emq_posteriors[label])
        if compare_value_func(emq_matrix) > compare_value_func(mle_matrix):
            emq_better.add(label)
    return emq_better


def get_contingency_matrix(labels, classifier_posteriors, emq_posteriors, quarter_y_arr):
    classifier_matrix = {'TPM': 0, 'FPM': 0, 'TNM': 0, 'FNM': 0}
    emq_matrix = {'TPM': 0, 'FPM': 0, 'TNM': 0, 'FNM': 0}
    for label in labels:
        temp_classifier_matrix = get_contingency_matrix_one_label(quarter_y_arr[label], classifier_posteriors[label])
        temp_emq_matrix = get_contingency_matrix_one_label(quarter_y_arr[label], emq_posteriors[label])
        for key, val in temp_classifier_matrix.items():
            classifier_matrix[key] += val
        for key, val in temp_emq_matrix.items():
            emq_matrix[key] += val
    return classifier_matrix, emq_matrix


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
    for i, c in enumerate(rcv1.target_names):
        quarter_y_arr[c] = np.asarray(rcv1.target[TEST_SET_START:TEST_SET_END, i].todense()).squeeze()
        full_y_arr[c] = np.asarray(rcv1.target[0:TEST_SET_END, i].todense()).squeeze()

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
    cr_set = set()
    cp_set = set()
    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)
        cr_set.add(cr)
        cp_set.add(cp)

    with open('./pickles/post_prob.pkl', 'rb') as f:
        classifier_posteriors = pickle.load(f)

    with open('./pickles/newemq_posteriors_0607.pkl', 'rb') as f:
        emq_posteriors = pickle.load(f)

    classifier_matrix, emq_matrix = get_contingency_matrix(labels, classifier_posteriors, emq_posteriors, quarter_y_arr)

    labels_len = len(labels)
    for key, val in classifier_matrix.items():
        classifier_matrix[key] = val / labels_len
        emq_matrix[key] = emq_matrix[key] / labels_len
    print("Classifier: ")
    pprint(classifier_matrix)
    print("\nEMQ: ")
    pprint(emq_matrix)

    classifier_accuracy = accuracy(classifier_matrix)
    emq_accuracy = accuracy(emq_matrix)
    print(f"Classifier accuracy: {classifier_accuracy}\nEMQ accuracy: {emq_accuracy}")

    print("##################################")
    print("Accuracy for Cr\n\n")

    classifier_matrix_cr, emq_matrix_cr = get_contingency_matrix(cr_set, classifier_posteriors, emq_posteriors, quarter_y_arr)
    classifier_matrix_cp, emq_matrix_cp = get_contingency_matrix(cp_set, classifier_posteriors, emq_posteriors, quarter_y_arr)

    cr_len = len(cr_set)
    for key, val in classifier_matrix_cr.items():
        classifier_matrix_cr[key] = val / cr_len
        emq_matrix_cr[key] = emq_matrix_cr[key] / cr_len
    print("Classifier CR: ")
    pprint(classifier_matrix_cr)
    print("EMQ: ")
    pprint(emq_matrix_cr)

    classifier_accuracy = accuracy(classifier_matrix_cr)
    emq_accuracy = accuracy(emq_matrix_cr)
    print(f"Classifier accuracy CR: {classifier_accuracy}\nEMQ accuracy CR: {emq_accuracy}")

    print("##################################")
    print("Accuracy for Cp\n\n")

    cp_len = len(cp_set)
    for key, val in classifier_matrix_cp.items():
        classifier_matrix_cp[key] = val / cp_len
        emq_matrix_cp[key] = emq_matrix_cp[key] / cp_len
    print("Classifier CP: ")
    pprint(classifier_matrix_cp)
    print("EMQ: ")
    pprint(emq_matrix_cp)
    classifier_accuracy = accuracy(classifier_matrix_cp)
    emq_accuracy = accuracy(emq_matrix_cp)
    print(f"Classifier accuracy CR: {classifier_accuracy}\nEMQ accuracy CR: {emq_accuracy}")


    # M131 CP {'TPM': 0.024793305506501846, 'FPM': 0.0014649221383849734, 'TNM': 0.9662766896773158, 'FNM': 0.007465082677797399}
    # 0.9910699951838177
    # M12 CR {'TPM': 0.02052897736394285, 'FPM': 0.003015130839621127, 'TNM': 0.9697282870444695, 'FNM': 0.006727604751966608}
    # 0.9902572644084123
    # M11 CR
    # {'TPM': 0.04687249157168085, 'FPM': 0.0034515973671536363, 'TNM': 0.9420051372611976, 'FNM': 0.007670773799967892}
    # 0.9888776288328784
    # ECAT CP
    # {'TPM': 0.0, 'FPM': 0.0, 'TNM': 0.8478086370203886, 'FNM': 0.1521913629796115}
    # 0.8478086370203886

    # ECAT class
    # {'TPM': 0.11099795312249157, 'FPM': 0.0142930245625301, 'TNM': 0.8335156124578584, 'FNM': 0.041193409857119924}
    # 0.94451356558035
    # M11
    # SAME
    # M12
    # SAME
    # M131
    # {'TPM': 0.025355193449991972, 'FPM': 0.001745866110130037, 'TNM': 0.9659957457055707, 'FNM': 0.006903194734307273}
    # 0.9913509391555627