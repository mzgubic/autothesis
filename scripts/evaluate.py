import matplotlib as plt
import pickle
from pathlib import Path

def plot_losses(loc):

    d = pickle.load(open(Path(loc)/'losses.pkl'), 'r')

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(losses))*every_n, d['training_loss'], label='training loss')
    ax.set_xlabel('training step')
    ax.set_ylabel('loss')
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    plt.savefig('Losses.pdf')
