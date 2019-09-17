import csv
import pickle
import argparse
import itertools
import time

from sklearn.datasets import fetch_rcv1

from cost_structure import *
from collections import namedtuple
from data_setup import TEST_SET_END, TEST_SET_START, get_setup_data
from utils import compute_prevalence


ContingencyTable = namedtuple('ContingencyTable', ['PP', 'PL', 'PW', 'LP', 'LL', 'LW', 'WP', 'WL', 'WW'])

def flatten(list_of_lists):
    """
    Flatten one level of nesting. See https://docs.python.org/3.6/library/itertools.html#recipes
    """
    return itertools.chain.from_iterable(list_of_lists)


def get_misclass_and_annot_costs(pairs, posterior_probabilities, y_arr, alphas, priors, cm, tau_rs, tau_ps):
    costs = Costs(cost_structure_1, pairs, posterior_probabilities, y_arr, alphas=alphas, prior_probabilities=priors)
    third_phase = costs.get_third_phase_costs(cm, tau_rs, tau_ps)
    return third_phase[2], third_phase[1]





parser = argparse.ArgumentParser(description="This program generates a tsv summary of a MineCore run.\n"
                                             "Every time MineCore's algorithm is ran, a tuple of four elements should be"
                                             " saved: (tau_rs, tau_ps, misclassification_cost_phase_2, misclassification_cost_phase_3)\n"
                                             "When given a file with the above structure as input, this program will generate "
                                             " for every label a csv table with the 9 contingency table values (P,L,W), the tau_rs and tau_ps "
                                             "costs, the misclassification and annotation costs, and the priors value on the test set.")

parser.add_argument('minecore_file_path', metavar='RUNFILE', help="Path to the MineCore run pickle data")
parser.add_argument('minecore_run_posteriors', metavar='POSTERIORFILE', help="Path to the MineCore run posteriors pickle data")
parser.add_argument('--compare-run-path', dest='compare_run_path', type=str, action='store', help="File path to another pickle. If given,"
                                                                                                  " the table will include "
                                                                                                  "misclassification and annotation costs "
                                                                                                  "for this file as well")
parser.add_argument('--compare-run-posteriors', dest='compare_run_posteriors', type=str, action='store', help="File path to the other MineCore run posteriors pickle."
                                                                                                              " If --compare-run-path is specified, "
                                                                                                              "this is mandatory.")

parser.add_argument('--alpha-values-path', dest='alpha_values_path', type=str, action='store', help="File path to the pickled alpha optimized values.")
parser.add_argument('--output-file', dest='output_file', type=str, action='store', default="out.tsv",help="Name of the output tsv file")

if __name__ == '__main__':
    args = parser.parse_args()
    posteriors_2, missclass_cost_2, annot_cost_2, alpha_values = None, None, None, None

    with open(args.minecore_file_path, 'rb') as f:
        tau_rs_1, tau_ps_1, cm_2_1, cm_3_1 = pickle.load(f)

    with open(args.minecore_run_posteriors, 'rb') as pf:
        posteriors_1 = pickle.load(pf)

    with open(args.alpha_values_path, 'rb') as alphaf:
        alpha_values = pickle.load(alphaf)
        alpha_values = dict(map(lambda kv: (kv[0], min(kv[1], key=kv[1].get)), alpha_values.items()))

    if args.compare_run_path and not args.compare_run_posteriors:
        print("When specifying --compare-run-path you need to specify --compare-run-posteriors as well, giving the path "
              "to the posterior probabilities pickle file")
        time.sleep(2)
        parser.print_usage()
        exit(1)

    if args.compare_run_path:
        with open(args.compare_run_path, 'rb') as f2:
            tau_rs_2, tau_ps_2, cm_2_2, cm_3_2 = pickle.load(f2)

        with open(args.compare_run_posteriors, 'rb') as pf2:
            posteriors_2 = pickle.load(pf2)

    rcv1 = fetch_rcv1()
    full_y_arr, quarter_y_arr, pairs, labels = get_setup_data(rcv1)
    tsv_file = open(args.output_file, 'w')
    writer = csv.writer(tsv_file, delimiter='\t', quotechar='|')
    writer.writerow(('Pair', 'PP', 'PL', 'PW', 'LP', 'LL', 'LW', 'WP', 'WL', 'WW', 'tau_r', 'tau_p', 'missclass_cost',
                    'annot_cost', 'missclass_cost_other', 'annot_cost_other', 'prior_cr', 'prior_cp'))


    priors, _ = compute_prevalence(labels, quarter_y_arr)
    missclass_cost_1, annot_cost_1 = get_misclass_and_annot_costs(pairs, posteriors_1, quarter_y_arr, alpha_values, priors, cm_3_1, tau_rs_1, tau_ps_1)
    if posteriors_2 is not None:
        missclass_cost_2, annot_cost_2 = get_misclass_and_annot_costs(pairs, posteriors_2, quarter_y_arr, None, None, cm_3_2, tau_rs_2, tau_ps_2)

    for pair, cont_vals in sorted(cm_3_1.items()):
        contingency_table = ContingencyTable._make(flatten(cont_vals))
        tau_rs_pair = tau_rs_1[pair]
        tau_ps_pair = tau_ps_1[pair]

        writer.writerow(
            (
                f"{pair[0]}-{pair[1]}",
                contingency_table.PP,
                contingency_table.PL,
                contingency_table.PW,
                contingency_table.LP,
                contingency_table.LL,
                contingency_table.LW,
                contingency_table.WP,
                contingency_table.WL,
                contingency_table.WW,
                tau_rs_pair,
                tau_ps_pair,
                missclass_cost_1[pair],
                annot_cost_1[pair],
                missclass_cost_2[pair] if missclass_cost_2 is not None else "N.D.",
                annot_cost_2[pair] if annot_cost_2 is not None else "N.D",
                "{:.3f}".format(priors[pair[0]]),
                "{:.3f}".format(priors[pair[1]])
            )
        )


    tsv_file.close()