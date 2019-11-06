import os
import time
import string
from pathlib import Path

data_path = Path('/data/atlassmallfiles/users/zgubic/thesis')
in_path = data_path / 'pdfs'
mid_path = data_path / 'raw_txts'
out_path = data_path / 'txts'


def generate_lines(path):
    
    with open(path, 'r') as handle:
        for line in handle:
            yield line

def has_3_or_more_spaces(line):
    if len(line.split()) > 3:
        return True
    else:
        return False

def is_not_toc(line):
    if '. . .' in line:
        return False
    else:
        return True

def has_alphabet_majority(line):
    N = len(line.replace(' ', ''))
    alphabet = list(string.ascii_letters)
    alpha_line = [x for x in line if x in alphabet]
    N_alpha = len(alpha_line)
    frac = N_alpha/N
    if frac > 0.5:
        return True
    else:
        # just "print(line)" causes the weirdest thing ever:
        # characters change appearance on screen, for example "\" becomes O with german umlaut...
        #print(' '.join(line.split()))
        return False

    return True

# convert to txts
for fname in os.listdir(in_path):
    
    # only look at pdfs
    if not Path(fname).suffix == '.pdf':
        continue
    out_fname = Path(fname).with_suffix('.txt')

    print(fname)
    print(out_fname)

    # if already done, don't bother
    if (mid_path/out_fname).exists():
        print('{} exists, not converting'.format(out_fname))
    else:
        # turn into text and remove empty lines
        os.system('pdftotext {} {}'.format(in_path/fname, mid_path/out_fname))

    # do the postprocessing
    with open(out_path/out_fname, 'w') as handle:

        try:
            for i, line in enumerate(generate_lines(mid_path/out_fname)):
                line = line.strip()

                keep = True
                keep = keep and has_3_or_more_spaces(line)
                keep = keep and is_not_toc(line)
                keep = keep and has_alphabet_majority(line)

                if not keep:
                    continue
                handle.write(line+'\n')

        except FileNotFoundError:
            continue

