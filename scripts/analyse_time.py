import os
import utils
import matplotlib.pyplot as plt
import itertools


def main():

    ks = ['cell']
    xs = ['hidden_size']

    for x, k in itertools.product(xs, ks):
        plot(x, k)


def plot(x, k):

    run_dir = utils.data_path / 'run' / 'character'
    dirs = [f for f in os.listdir(run_dir) if os.path.isdir(run_dir/f)]

    # extract times
    times = {}
    for d in dirs:
        components = d.split('__')
        this_x = float([s for s in components if x in s][0].replace(x, ''))
        this_kind = [s for s in components]
        this_kind = [s for s in components if k in s][0].replace(k, '')

        model_dir = run_dir / d
 
        try:
            with open(model_dir / 'time.txt', 'r') as handle:
                t = float(handle.readline())
            times[this_kind][this_x] = t

        except KeyError:
            times[this_kind] = {this_x:t}

        except FileNotFoundError:
            pass

    # prepare plots
    kinds = list(times.keys())
    xs = {kind:sorted(list(times[kind].keys())) for kind in kinds}
    ys = {kind:[times[kind][x] for x in xs[kind]] for kind in kinds}

    # plot
    fig, ax = plt.subplots()
    for kind in kinds:
        ax.plot(xs[kind], ys[kind], label=kind)
        ax.set_xlabel(x)
        ax.set_ylabel('training time (h)')
    ax.legend()
    plt.savefig('../figures/training_time_vs_{}_for_{}.pdf'.format(x, k))


if __name__ == '__main__':
    main()
