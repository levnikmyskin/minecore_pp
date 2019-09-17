import matplotlib.pyplot as plt
import matplotlib.lines as lines
from sklearn.datasets import fetch_rcv1
from decimal import Decimal, ROUND_05UP
import pickle
import numpy as np
import csv
from cost_structure import Costs, cost_structure_1
from emq_attempt import emq_attempt
from minecore import MineCore
from utils import compute_prevalence
from similarity_check import absolute_error
from postemq_analysis import get_contingency_matrix_one_label, get_emq_better_posteriors, accuracy, precision, \
    has_no_positives
from alpha_analysis import contingency_matrix_mle_alpha, normalize_contingency_table


def round_with_decimal_precision(value: float, quantize_str='.1'):
    return Decimal(Decimal(value).quantize(Decimal(quantize_str), rounding=ROUND_05UP))


def show_delta_costs_graph(cost_1, cost_2, pairs=None, x_label="Pairs", y_label="Cost difference"):
    overall_diff_ratio = 0
    worse_count = 0
    improve_count = 0

    keys = pairs if pairs is not None else cost_2.keys()
    to_plot = list()
    for key in keys:
        d = cost_1[key] / cost_2[key]
        overall_diff_ratio += d
        # percent_diff = d * 100 / cost_1[key]
        # diff[key] = percent_diff
        if d > 1:
            improve_count += 1
        else:
            worse_count += 1
        to_plot.append((f"{key[0]}-{key[1]}", d))

    print(f"Diff mean: {overall_diff_ratio / (improve_count + worse_count)}")
    to_plot = sorted(to_plot,key=lambda x:x[1])

    plt.figure(figsize=(20,10))

    for key,diff in to_plot:
        plt.plot(key, diff, 'go')

    keys = list(map(lambda k: f"{k[0]}-{k[1]}", keys))

    worse_percent = worse_count / len(keys)
    improve_percent = improve_count / len(keys)

    plt.plot(keys, [1 for _ in range(len(keys))], 'r-')
    plt.xticks(rotation='vertical')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.annotate(f"Improved: {int(improve_percent * 100)}%; Worsened: {int(worse_percent * 100)}%", xy=(0.5, 0),
                 xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14,
                 ha='center', va='bottom')
    plt.show()


def order_by_distribution_drift(labels, training_y_arr, quarter_y_arr):
    pos_training_prev, neg_training_prev = compute_prevalence(labels, training_y_arr)
    pos_test_prev, neg_test_prev = compute_prevalence(labels, quarter_y_arr)
    prevalence_drift = dict()
    for label in labels:
        prevalence_drift[label] = absolute_error(pos_training_prev[label], neg_training_prev[label], pos_test_prev[label], neg_test_prev[label])

    return sorted(pairs, key=lambda t: (prevalence_drift[t[0]] + prevalence_drift[t[1]]) / 2), prevalence_drift


def show_graph_deltaacc_priors(bef_emq_priors, after_emq_priors, bef_costs, after_costs, x_label="Delta accuracy priors", y_label="Cost difference"):
    pos_training_prev, neg_training_prev = compute_prevalence(labels, quarter_y_arr)
    delta_accuracy = dict()
    for label in labels:
        true_priors = np.array([neg_training_prev[label], pos_training_prev[label]])
        before_diff = np.abs(np.array(bef_emq_priors[label]) - true_priors).mean()
        after_diff = np.abs(np.array(after_emq_priors[label]) - true_priors).mean()
        delta_accuracy[label] = after_diff - before_diff

    for key in after_costs.keys():
        d = after_costs[key] - bef_costs[key]
        acc_mean = delta_accuracy[key[0]] + delta_accuracy[key[1]]
        plt.plot(acc_mean, d, 'go')

    plt.plot(np.linspace(-0.04, 0.02), [0 for _ in range(50)], 'r-')
    plt.plot([0 for _ in range(50)], np.linspace(-5000, 16000), 'r-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_graph_deltacc_posteriors(classifier_posteriors, emq_posteriors, bef_costs, after_costs, x_label="Delta accuracy posteriors", y_label="Cost difference"):
    classifier_accuracies = dict()
    emq_accuracies = dict()
    for label in labels:
        classifier_matrix = get_contingency_matrix_one_label(quarter_y_arr[label], classifier_posteriors[label])
        emq_matrix = get_contingency_matrix_one_label(quarter_y_arr[label], emq_posteriors[label])
        classifier_accuracies[label] = (classifier_matrix['TPM'] + classifier_matrix['TNM']) / sum(classifier_matrix.values())
        emq_accuracies[label] = (emq_matrix['TPM'] + emq_matrix['TNM']) / sum(emq_matrix.values())

    for key in after_costs.keys():
        d = after_costs[key] - bef_costs[key]
        acc_mean = (emq_accuracies[key[0]] + emq_accuracies[key[1]]) - (classifier_accuracies[key[0]] + classifier_accuracies[key[1]])
        plt.plot(acc_mean, d, 'go')

    plt.plot(np.linspace(-0.0016, 0.0006), [0 for _ in range(50)], 'r-')
    plt.plot([0 for _ in range(50)], np.linspace(-5000, 16000), 'r-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_graph_alpha_class(cost_1, cost_2, pairs, alphas, alpha_index):
    keys = pairs if pairs is not None else cost_2.keys()
    to_plot = dict()
    # to_plot = list()
    alphas = dict(map(lambda item: (item[0], (round_with_decimal_precision(item[1][0]), round_with_decimal_precision(item[1][1]))), alphas.items()))
    for key in keys:
        d = cost_1[key] / cost_2[key]
        index = f"{alphas[key][0]}-{alphas[key][1]}"
        # to_plot.append((index, d))
        val = to_plot.setdefault(index, (0, 0))
        to_plot[index] = (val[0] + 1 if d > 1 else val[0], val[1] + 1)

    # Transform dict into list of tuples and sort it
    to_plot = sorted(map(lambda i: (i[0], i[1]), filter(lambda i: i[1][1] > 1, to_plot.items())), key=lambda x:x[1][0] / x[1][1])
    # to_plot = sorted(to_plot,key=lambda x:x[1])

    plt.figure(figsize=(20,10))

    for key,diff in to_plot:
        plt.bar(f"{key} ($n={diff[1]}$)", diff[0] / diff[1])


    # alphas = list(map(lambda a: f"{a[1][0]}-{a[1][1]}", alphas.items()))
    # plt.plot(alphas, [1 for _ in range(len(alphas))], 'r-')
    plt.xticks(rotation='vertical')
    plt.xlabel("Alphas")
    plt.ylabel("Percentage of improved pairs")
    plt.show()


def save_alpha_data_csv(csv_name, cost_1, cost_2, pairs, alphas, round_alphas=False):
    csv_file = open(csv_name, 'w')
    csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='|')
    csv_writer.writerow(["Pair", "alphar-p", "alpha_r", "alpha_p", "delta_cost", "improved"])
    if round_alphas:
        alphas = dict(map(lambda item: (item[0], (round_with_decimal_precision(item[1][0]), round_with_decimal_precision(item[1][1]))), alphas.items()))
    for pair in pairs:
        d = cost_1[pair] / cost_2[pair]
        alpha = alphas[pair]
        if d >= 1:
            csv_writer.writerow([pair, f"{str(alpha[0])}-{str(alpha[1])}", alpha[0], alpha[1], d, "Improved"])
        else:
            csv_writer.writerow([pair, f"{str(alpha[0])}-{str(alpha[1])}", alpha[0], alpha[1], d, "Deteriorated"])
    csv_file.close()

def __bold_text(text):
    return r"$\mathbf{" + text + r"}$"


def show_nopositives_graph(cost_1, cost_2, pairs, posteriors, alphas, minecore):
    to_plot = list()
    mle_matrices, alpha_matrices = dict(), dict()
    for pair in pairs:
        d = cost_1[pair] / cost_2[pair]
        mle_matr_cr, alpha_matr_cr = contingency_matrix_mle_alpha(posteriors, pair, alphas, minecore, cr=True)
        mle_matr_cp, alpha_matr_cp = contingency_matrix_mle_alpha(posteriors, pair, alphas, minecore, cr=False)
        if has_no_positives(mle_matr_cr) or has_no_positives(mle_matr_cp):
            mle_matrices[pair] = [mle_matr_cr, mle_matr_cp, d]
        if has_no_positives(alpha_matr_cr) and has_no_positives(alpha_matr_cp):
            alpha_matrices[pair] = [alpha_matr_cr, alpha_matr_cp, d]
            to_plot.append((f"{__bold_text(pair[0])}-{__bold_text(pair[1])}", d))
        elif has_no_positives(alpha_matr_cr):
            alpha_matrices[pair] = [alpha_matr_cr, alpha_matr_cp, d]
            to_plot.append((f"{__bold_text(pair[0])}-{pair[1]}", d))
        elif has_no_positives(alpha_matr_cp):
            alpha_matrices[pair] = [alpha_matr_cr, alpha_matr_cp, d]
            to_plot.append((f"{pair[0]}-{__bold_text(pair[1])}", d))

    print(f"Number of alpha pairs with no positives in either cr or cp: {len(alpha_matrices)}\n"
          f"Number of mle pairs // //: {len(mle_matrices)}\n"
          f"Improved alpha pairs // //: {len(list(filter(lambda el: el[2] > 1, alpha_matrices.values())))}\n"
          f"Improved mle pairs // //: {len(list(filter(lambda el: el[2] < 1, mle_matrices.values())))}\n"
          f"Number of no cr positives: {len(list(filter(lambda el: has_no_positives(el[0]), alpha_matrices.values())))}\n"
          f"Number of no cp positives: {len(list(filter(lambda el: has_no_positives(el[1]), alpha_matrices.values())))}")

    to_plot = sorted(to_plot,key=lambda x:x[1])
    plt.figure(figsize=(15,7))

    for key,diff in to_plot:
        plt.plot(key, diff, 'go')

    keys = list(k[0] for k in to_plot)

    plt.plot(keys, [1 for _ in range(len(keys))], 'r-')
    plt.xlabel("Pairs with no positives for either $c_r$ or $c_p$ (Pr$^{u}$)")
    plt.ylabel("Cost difference")
    plt.show()

    """
    Number of alpha pairs with no positives in either cr or cp: 7
    Number of mle pairs // //: 0
    Improved alpha pairs // //: 5
    Improved mle pairs // //: 0
    Number of no cr positives: 0
    Number of no cp positives: 7
    """


def show_nopositives_graph_emq(cost_1, cost_2, pairs, true_posteriors, class_posteriors, emq_posteriors):
    to_plot = list()
    mle_matrices, emq_matrices = dict(), dict()
    for pair in pairs:
        d = cost_1[pair] / cost_2[pair]
        mle_matr_cr = get_contingency_matrix_one_label(true_posteriors[pair[0]], class_posteriors[pair[0]])
        emq_matr_cr = get_contingency_matrix_one_label(true_posteriors[pair[0]], emq_posteriors[pair[0]])
        mle_matr_cp = get_contingency_matrix_one_label(true_posteriors[pair[1]], class_posteriors[pair[1]])
        emq_matr_cp = get_contingency_matrix_one_label(true_posteriors[pair[1]], emq_posteriors[pair[1]])

        if has_no_positives(mle_matr_cr) or has_no_positives(mle_matr_cp):
            mle_matrices[pair] = [mle_matr_cr, mle_matr_cp, d]
        if has_no_positives(emq_matr_cr) and has_no_positives(emq_matr_cp):
            emq_matrices[pair] = [emq_matr_cr, emq_matr_cp, d]
            to_plot.append((f"{__bold_text(pair[0])}-{__bold_text(pair[1])}", d))
        elif has_no_positives(emq_matr_cr):
            emq_matrices[pair] = [emq_matr_cr, emq_matr_cp, d]
            to_plot.append((f"{__bold_text(pair[0])}-{pair[1]}", d))
        elif has_no_positives(emq_matr_cp):
            emq_matrices[pair] = [emq_matr_cr, emq_matr_cp, d]
            to_plot.append((f"{pair[0]}-{__bold_text(pair[1])}", d))

    print(f"Number of emq pairs with no positives in either cr or cp: {len(emq_matrices)}\n"
          f"Number of mle pairs // //: {len(mle_matrices)}\n"
          f"Improved emq pairs // //: {len(list(filter(lambda el: el[2] > 1, emq_matrices.values())))}\n"
          f"Improved mle pairs // //: {len(list(filter(lambda el: el[2] < 1, mle_matrices.values())))}\n"
          f"Number of no cr positives: {len(list(filter(lambda el: has_no_positives(el[0]), emq_matrices.values())))}\n"
          f"Number of no cp positives: {len(list(filter(lambda el: has_no_positives(el[1]), emq_matrices.values())))}")

    to_plot = sorted(to_plot,key=lambda x:x[1])
    plt.figure(figsize=(20,10))

    for key,diff in to_plot:
        plt.plot(key, diff, 'go')

    keys = list(k[0] for k in to_plot)

    plt.plot(keys, [1 for _ in range(len(keys))], 'r-')
    plt.xticks(rotation='vertical')
    plt.xlabel("Pairs with no positives for either $c_r$ or $c_p$ (Pr$^{EMQ}$)")
    plt.ylabel("Cost difference")
    plt.show()

    """
    Number of emq pairs with no positives in either cr or cp: 92
    Number of mle pairs // //: 0
    Improved emq pairs // //: 7
    Improved mle pairs // //: 0
    Number of no cr positives: 70
    Number of no cp positives: 46
    """


def show_tp_deltacosts_graph(cost_1, cost_2, pairs, posteriors, alphas, minecore):
    to_plot = dict()
    for pair in pairs:
        d = cost_1[pair] / cost_2[pair]
        mle_matr, alpha_matr = contingency_matrix_mle_alpha(posteriors, pair, alphas, minecore, cr=True)
        mle_matr = normalize_contingency_table(mle_matr, 199328)
        alpha_matr = normalize_contingency_table(alpha_matr, 199328)
        mle_matr_cp, alpha_matr_cp = contingency_matrix_mle_alpha(posteriors, pair, alphas, minecore, cr=False)
        mle_matr_cp = normalize_contingency_table(mle_matr_cp, 199328)
        alpha_matr_cp = normalize_contingency_table(alpha_matr_cp, 199328)
        precision_cr = precision(alpha_matr, "TPM", "FPM")
        precision_cp = precision(alpha_matr_cp, "TPM", "FPM")
        rounded_tp = round_with_decimal_precision(precision_cp, quantize_str='.1')
        rounded_tr = round_with_decimal_precision(precision_cr, quantize_str='.1')
        val = to_plot.setdefault(f"{rounded_tr}-{rounded_tp}", (0, 0))
        to_plot[f"{rounded_tr}-{rounded_tp}"] = (val[0] + d if d > 1 else val[0] - d, val[1] + 1)

    # to_plot = sorted(to_plot,key=lambda x:x[1])
    to_plot = sorted(map(lambda i: (i[0], i[1]), filter(lambda i: i[1][1] > 1,to_plot.items())), key=lambda x:x[1][0] / x[1][1])
    print(to_plot)
    plt.figure(figsize=(15, 7))

    for key,diff in to_plot:
        plt.bar(f"{key} ($n={diff[1]}$)", diff[0] / diff[1], width=0.3)

    plt.xlabel("TPM Precision $\{c_r, c_p\}$")
    plt.ylabel("Average cost gain")
    plt.show()


def show_notpnofp_deltacosts_graph(cost_1, cost_2, pairs, emq_posteriors, true_posteriors):
    to_plot = dict()
    for pair in pairs:
        d = cost_1[pair] / cost_2[pair]
        emq_matr_cr = normalize_contingency_table(get_contingency_matrix_one_label(true_posteriors[pair[0]], emq_posteriors[pair[0]]), 199328)
        emq_matr_cp = normalize_contingency_table(get_contingency_matrix_one_label(true_posteriors[pair[1]], emq_posteriors[pair[1]]), 199328)
        if has_no_positives(emq_matr_cr) and has_no_positives(emq_matr_cp):
            key = "No pos $\{c_r, c_p\}$"
        elif has_no_positives(emq_matr_cr):
            key = "No pos $c_r$"
        elif has_no_positives(emq_matr_cp):
            key = "No pos $c_p$"
        else: continue
        val = to_plot.setdefault(key, (0, 0))
        to_plot[key] = (val[0] + d if d > 1 else val[0] - d, val[1] + 1)

    to_plot = sorted(map(lambda i: (i[0], i[1]), filter(lambda i: i[1][1] > 1,to_plot.items())), key=lambda x:x[1][0] / x[1][1])
    print(to_plot)
    plt.figure(figsize=(15, 7))

    for key,diff in to_plot:
        plt.bar(f"{key} ($n={diff[1]}$)", diff[0] / diff[1], width=0.3)

    plt.xlabel("Pairs with no positives")
    plt.ylabel("Average cost gain")
    plt.show()

def show_risk_deltacosts_graph(cost_1, cost_2, pairs, posteriors, alphas, minecore):
    to_plot = list()
    for pair in pairs:
        alpha_r, alpha_p = alphas[pair]
        us_r = minecore.user_posteriors(alpha_r)[pair[0]]
        us_p = minecore.user_posteriors(alpha_p)[pair[1]]
        d = cost_1[pair] / cost_2[pair]
        risk_us = minecore.get_h_risk_h(us_r[:, 1], us_p[:, 1], costs.cost_matrix, need_risk=True)[1]
        risk_s = minecore.get_h_risk_h(posteriors[pair[0]][:, 1], posteriors[pair[1]][:, 1], costs.cost_matrix, need_risk=True)[1]
        to_plot.append((abs(risk_us.mean() - risk_s.mean()), d))

    to_plot = sorted(to_plot,key=lambda x:x[1])
    plt.figure(figsize=(20,10))

    for key,diff in to_plot:
        plt.plot(key, diff, 'go')

    plt.xlabel("Risk us")
    plt.ylabel("Cost difference")
    plt.show()



def show_automatic_costs():
    before_costs = Costs(cost_structure_1, pairs, classifier_posteriors, quarter_y_arr)
    before_emq_costs = before_costs.get_automatic_costs()

    after_costs = Costs(cost_structure_1, pairs, emq_posteriors, quarter_y_arr)
    after_emq_costs = after_costs.get_automatic_costs()
    show_delta_costs_graph(before_emq_costs[0], after_emq_costs[0])

    print(f"Before EMQ Costs: {before_emq_costs}\nAfter EMQ Costs: {after_emq_costs}")


def show_delta_costs_with_labeled_points(cost_1, cost_2, emq_better_set, compared_metric):
    worse_count = 0
    improve_count = 0
    overall_diff_ratio = 0

    keys = pairs if pairs is not None else cost_2.keys()
    to_plot = list()
    for key in keys:
        d = cost_1[key] / cost_2[key]
        overall_diff_ratio += d
        if d > 1:
            improve_count += 1
        else:
            worse_count += 1

        c_r, c_p = key
        if c_r in emq_better_set and c_p in emq_better_set:
            color = "ro"
        elif c_r in emq_better_set:
            color = "bo"
        elif c_p in emq_better_set:
            color = "yo"
        else:
            color = "go"
        to_plot.append((f"{key[0]}-{key[1]}", d, color))

    keys = list(map(lambda k: f"{k[0]}-{k[1]}", keys))

    worse_percent = worse_count / len(keys)
    improve_percent = improve_count / len(keys)

    to_plot = sorted(to_plot,key=lambda x:x[1])
    plt.figure(figsize=(20,10))

    for key,diff, color in to_plot:
        plt.plot(key, diff, color)

    plt.plot(keys, [1 for _ in range(len(keys))], 'r-')
    plt.xticks(rotation='vertical')
    plt.xlabel("Pairs")
    plt.ylabel("Cost difference")
    plt.annotate(f"Improved: {int(improve_percent * 100)}%; Worsened: {int(worse_percent * 100)}%", xy=(0.5, 0),
                 xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14,
                 ha='center', va='bottom')
    green_patch = lines.Line2D([], [], marker="o", color="green", label=f"Red: EMQ {compared_metric} > for cr and cp")
    blue_patch = lines.Line2D([], [], marker="o", color="blue", label=f"Blue: EMQ {compared_metric} > for cr only")
    yellow_patch = lines.Line2D([], [], marker="o", color="yellow", label=f"Yellow: EMQ {compared_metric} > for cp only")
    red_patch = lines.Line2D([], [], marker="o", color="red", label=f"Green: EMQ {compared_metric} < for cr and cp")
    plt.legend(handles=(green_patch, blue_patch, yellow_patch, red_patch))
    # plt.legend(handles=(green_patch, red_patch))
    plt.show()


def show_distribution_drift_graph(train_prev, test_prev, labels):
    to_plot = list()
    for label in labels:
        to_plot.append((label, abs(train_prev[label] - test_prev[label])))

    to_plot = sorted(to_plot, key=lambda x: x[1])
    for label, diff in to_plot:
        plt.plot(label, diff, 'go')

    plt.ylim(bottom=0)
    plt.xticks(rotation='vertical')
    plt.xlabel("Labels")
    plt.ylabel("Train-Test shift")
    plt.show()


def show_graphs(before_costs, after_costs):
    sorted_pairs, prev_drift = order_by_distribution_drift()
    drift_mean = sum(prev_drift.values()) / len(prev_drift)
    print(f"Sorted pairs: {sorted_pairs}\n"
          f"Drift mean: {drift_mean}\n"
          f"Lowest drift: cr={prev_drift[sorted_pairs[0][0]]}; cp={prev_drift[sorted_pairs[0][1]]}\n"
          f"Highest drift: cr={prev_drift[sorted_pairs[-1][0]]}; cp={prev_drift[sorted_pairs[-1][1]]}")

    show_delta_costs_graph(before_costs, after_costs, pairs, y_label="Cost difference (second phase)")
    show_graph_deltaacc_priors(befemq_priors, aft_priors, before_costs, after_costs, y_label="Cost difference (second phase)")
    show_graph_deltacc_posteriors(classifier_posteriors, emq_posteriors, before_costs, after_costs, y_label="Cost difference (second_phase)")


if __name__ == '__main__':

    pairs = [('M12', 'M14'), ('M12', 'CCAT'), ('M12', 'M132'), ('M12', 'E21'), ('M12', 'M131'), ('M132', 'GPOL'),
             ('M132', 'CCAT'), ('M132', 'M12'), ('M132', 'M131'), ('M132', 'GCAT'), ('M131', 'CCAT'), ('M131', 'M132'),
             ('M131', 'E12'), ('M131', 'ECAT'), ('M131', 'M12'), ('E12', 'M11'), ('E12', 'GDIP'), ('E12', 'E212'),
             ('E12', 'M131'), ('E12', 'E21'), ('C21', 'C17'), ('C21', 'C15'), ('C21', 'ECAT'), ('C21', 'C31'),
             ('C21', 'M141'), ('E212', 'GPOL'), ('E212', 'E12'), ('E212', 'M12'), ('E212', 'MCAT'), ('E212', 'C17'),
             ('GCRIM', 'E212'), ('GCRIM', 'C15'), ('GCRIM', 'C18'), ('GCRIM', 'GDIP'), ('GCRIM', 'GPOL'), ('C24', 'GDIP'),
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

    with open('./pickles/minecore_1309_rorp_050_99_alphas.pkl', 'rb') as f:
        res_before = pickle.load(f)

    with open('./pickles/minecorepp_1309_rorp_050_99_alphas.pkl', 'rb') as f:
        res_after = pickle.load(f)

    # with open('./pickles/before_emq.pkl', 'rb') as f:
    #     res_before = pickle.load(f)

    # with open('./pickles/bef_emq_priors.pkl', 'rb') as f:
    #     befemq_priors = pickle.load(f)
    #
    # with open('./pickles/emq_priors.pkl', 'rb') as f:
    #     aft_priors = pickle.load(f)
    #
    with open('./pickles/post_prob.pkl', 'rb') as f:
        classifier_posteriors = pickle.load(f)
    #
    with open('./pickles/newemq_posteriors_0607.pkl', 'rb') as f:
        emq_posteriors = pickle.load(f)
    with open('./pickles/alpha_dict_labels_3108.pkl', 'rb') as f:
        alphas = pickle.load(f)

    alphas = dict(map(lambda kv: (kv[0], min(kv[1], key=kv[1].get)), alphas.items()))
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

    labels = set()
    for cr, cp in pairs:
        labels.add(cr)
        labels.add(cp)

    prior_probabilities, neg_priors = compute_prevalence(labels, training_y_arr)
    ro_r = 0.50
    ro_p = 0.99
    # alphas = dict(map(lambda kv: (kv[0], (1.0, 1.0)), alphas.items()))
    costs = Costs(cost_structure_1, pairs, classifier_posteriors, quarter_y_arr, alphas=alphas,
                  prior_probabilities=prior_probabilities, ro_r=ro_r, ro_p=ro_p)
    minecore = MineCore(pairs, prior_probabilities, classifier_posteriors, quarter_y_arr, alphas, ro_r, ro_p)

    costs_results_after = costs.get_third_phase_costs(res_after[3], res_after[0], res_after[1])
    costs_results_before = costs.get_third_phase_costs(res_before[3], res_before[0], res_before[1])
    overall_costs_after = costs_results_after[0]
    overall_costs_before = costs_results_before[0]

    # show_graphs(overall_costs_before, overall_costs_after)
    show_delta_costs_graph(overall_costs_before, overall_costs_after, pairs)
    # show_notpnofp_deltacosts_graph(overall_costs_before, overall_costs_after, pairs, emq_posteriors, quarter_y_arr)
    # show_nopositives_graph(overall_costs_before, overall_costs_after, pairs, classifier_posteriors, alphas, minecore)
    # show_nopositives_graph_emq(overall_costs_before, overall_costs_after, pairs, quarter_y_arr, classifier_posteriors, emq_posteriors)
    # show_graph_alpha_class(overall_costs_before, overall_costs_after, pairs, alphas, 0)
    # save_alpha_data_csv("alphas_data-mc0308.tsv", overall_costs_before, overall_costs_after, pairs, alphas, round_alphas=True)
    # show_tp_deltacosts_graph(overall_costs_before, overall_costs_after, pairs, classifier_posteriors, alphas, minecore)
    # show_risk_deltacosts_graph(overall_costs_before, overall_costs_after, pairs, classifier_posteriors, alphas, minecore)

    # show_delta_costs_with_labeled_points(
    #     overall_costs_before, overall_costs_after,
    #     get_emq_better_posteriors(labels, quarter_y_arr, classifier_posteriors, emq_posteriors, accuracy), "accuracy"
    # )
