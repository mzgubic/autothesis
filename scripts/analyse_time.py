import os
import utils
import matplotlib.pyplot as plt


def main():

    common_before = 'cellRNN__hidden_size64__learning_rate0.001__batch_size64__max_len20__n_cores'
    common_after = '__n_steps100'

    times = {}
    for n_cores in range(1, 16+1):
        model_dir = utils.data_path / 'run' / 'character' / '{}{}{}'.format(common_before, str(n_cores), common_after)

        try:
            with open(model_dir / 'time.txt', 'r') as handle:
                t = float(handle.readline())
                times[n_cores] = t

        except FileNotFoundError:
            pass

    cores = sorted(list(times.keys()))
    speeds = [1/times[c] for c in cores]

    fig, ax = plt.subplots()
    ax.plot(cores, speeds)
    ax.set_xlabel('number of cores')
    ax.set_xlabel('1/training time (1/s)')
    plt.savefig('timetest.pdf')



        





if __name__ == '__main__':
    main()
