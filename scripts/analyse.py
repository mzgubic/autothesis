import os
import utils
import matplotlib.pyplot as plt
import itertools


def main():

    # show 
    x = 'hidden_size'
    kind = 'cell'
    cut = 'max_len'
    cut_vals = [10, 20, 50, 100]
    for cut_val in cut_vals:
        plot_time(x, kind, cut, cut_val)
        plot_best_accuracies(x, kind, cut, cut_val)

    x = 'hidden_size'
    kind = 'max_len'
    cut = 'cell'
    cut_vals = ['RNN', 'GRU', 'LSTM']
    for cut_val in cut_vals:
        plot_time(x, kind, cut, cut_val)
        plot_best_accuracies(x, kind, cut, cut_val)




def extract_values(x, y, k, cut, cut_val):

    run_dir = utils.data_path / 'run' / 'character'
    dirs = [f for f in os.listdir(run_dir) if os.path.isdir(run_dir/f)]

    # extract y values
    ys = {}
    for d in dirs:
        # determine the values
        components = d.split('__')
        this_x = float([s for s in components if x in s][0].replace(x, ''))
        this_kind = [s for s in components]
        this_kind = [s for s in components if k in s][0].replace(k, '')

        # apply the cut
        try:
            keep = float([s for s in components if cut in s][0].replace(cut, '')) == float(cut_val)
        except ValueError:
            keep = [s for s in components if cut in s][0].replace(cut, '') == cut_val
        if not keep:
            continue

        model_dir = run_dir / d
 
        try:
            with open(model_dir / '{}.txt'.format(y), 'r') as handle:
                y_val = float(handle.readline())

            ys[this_kind][this_x] = y_val

        except KeyError:
            ys[this_kind] = {this_x:y_val}

        except FileNotFoundError:
            pass

    return ys

def plot_time(x, k, cut, cut_val):

    # extract values
    times = extract_values(x, 'time', k, cut, cut_val)

    # prepare plots
    kinds = list(times.keys())
    xs = {kind:sorted(list(times[kind].keys())) for kind in kinds}
    ys = {kind:[times[kind][x] for x in xs[kind]] for kind in kinds}

    # plot
    fig, ax = plt.subplots()
    for kind in kinds:
        ax.plot(xs[kind], ys[kind], label='{}={}'.format(k, kind))
        ax.set_xlabel(x)
        ax.set_ylabel('training time (h)')
    ax.set_title('{} == {}'.format(cut, cut_val))
    ax.legend()
    plt.savefig('../figures/training_time_vs_{}_for_{}_at_{}{}.pdf'.format(x, k, cut, cut_val))

def plot_best_accuracies(x, k, cut, cut_val):

    # extract accuracies
    splits = ['train', 'valid']
    accuracies = {s:extract_values(x, 'best_{}_acc'.format(s), k, cut, cut_val) for s in splits}

    # prepare the plot
    fig, ax = plt.subplots()
    for split, linestyle in zip(splits, [':', '-']):

        # prepare plots
        acc = accuracies[split]
        kinds = list(acc.keys())
        xs = {kind:sorted(list(acc[kind].keys())) for kind in kinds}
        ys = {kind:[acc[kind][x] for x in xs[kind]] for kind in kinds}

        for kind in kinds:
            ax.plot(xs[kind], ys[kind], linestyle, label='{}={} ({})'.format(k, kind, split))
            ax.set_xlabel(x)
            ax.set_ylabel('accuracy')
        ax.set_title('{} == {}'.format(cut, cut_val))
        ax.legend()
        plt.savefig('../figures/accuracy_vs_{}_for_{}_at_{}{}.pdf'.format(x, k, cut, cut_val))

        # reset the color cycle
        plt.gca().set_prop_cycle(None)



if __name__ == '__main__':
    main()
