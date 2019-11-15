import argparse
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def plot_losses(loc):

    # load data
    model_dir = Path(loc)
    loss = pickle.load(open(model_dir/'losses.pkl', 'rb'))
    settings = pickle.load(open(model_dir/'settings.pkl', 'rb'))

    # prepare
    every_n = settings['every_n']

    # plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(loss['train']))*every_n, loss['train'], label='training loss')
    ax.plot(np.arange(len(loss['valid']))*every_n, loss['valid'], label='validation loss')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.savefig(model_dir/'Losses.pdf')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='/data/atlassmallfiles/users/zgubic/thesis/run/character/CharacterRNN_20_100steps')
    args = parser.parse_args()

    plot_losses(args.input_dir)
