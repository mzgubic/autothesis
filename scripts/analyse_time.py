import os
import utils
import matplotlib.pyplot as plt


def main():

    common_before = 'cellRNN__hidden_size64__learning_rate0.001__batch_size64__max_len20__n_cores'
    common_after = '__n_steps100000'

    times = {}
    for n_cores in range(1, 16+1):
        model_dir = utils.data_path / 'run' / 'character' / '{}{}{}'.format(common_before, str(n_cores), common_after)

        try:
            with open(model_dir / 'time.txt', 'r') as handle:
                t = float(handle.readline())/60.
                times[n_cores] = t
                print(n_cores, t)

        except FileNotFoundError:
            pass

    cores = sorted(list(times.keys()))
    speeds = [1/times[c] for c in cores]

    fig, ax = plt.subplots()
    ax.plot(cores, speeds)
    ax.set_xlabel('number of cores')
    ax.set_ylabel('1/training time (1/min)')
    plt.savefig('../figures/speed.pdf')

    fig, ax = plt.subplots()
    ax.plot(cores, [times[c] for c in cores])
    ax.set_xlabel('number of cores')
    ax.set_ylabel('training time (min)')
    plt.savefig('../figures/time.pdf')



        





if __name__ == '__main__':
    main()
