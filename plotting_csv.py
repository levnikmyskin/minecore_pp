import matplotlib.pyplot as plt
import csv
from collections import namedtuple

CsvData = namedtuple('CsvData', ['pair', 'pp', 'pl', 'pw', 'lp', 'll', 'lw', 'wp', 'wl', 'ww',
                                 'tau_r', 'tau_p', 'missclass', 'annot', 'missclass_other', 'annot_other',
                                 'prior_cr', 'prior_cp'])


def get_x_and_y_axis(to_plot):
    return list(map(lambda t: t[0], to_plot)), list(map(lambda t: t[1], to_plot))

if __name__ == '__main__':
    plt.figure(figsize=(20,10))
    to_plot_1 = list()
    to_plot_2 = list()
    with open('out.tsv', newline='') as tsvf:
        reader = csv.reader(tsvf, delimiter="\t", quotechar='|')
        next(reader)  # skip title
        for row in reader:
            data = CsvData._make(row)
            to_plot_1.append((float(data.prior_cr), float(data.annot)))
            #to_plot_2.append((float(data.prior_cr) + float(data.prior_cp), float(data.missclass_other)))

    to_plot_1.sort()
    #to_plot_2.sort()
    x_axis, y_axis = get_x_and_y_axis(to_plot_1)
    plt.plot(x_axis, y_axis, 'r-')
    #x_axis_2, y_axis_2 = get_x_and_y_axis(to_plot_2)
    # plt.plot(x_axis_2, y_axis_2, 'g-')
    plt.ylabel("Missclassification costs")
    plt.xlabel("Prior cr + Prior cp")
    plt.show()

