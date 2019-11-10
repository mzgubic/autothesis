import os
import time
import string
from pathlib import Path
import nltk
import utils
nltk.download('words')

in_path = utils.data_path / 'pdfs'
mid_path = utils.data_path / 'raw_txts'
out_path = utils.data_path / 'txts'


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

def remove_failed_transcriptions(fname, out_fname):
    
    size = os.path.getsize(out_path / out_fname)
    if size < 1000:
        os.system('rm {} {} {}'.format(in_path/fname, mid_path/out_fname, out_path/out_fname))

def remove_nonenglish_theses(fname, out_fname, english_words):

    try:
        with open(out_path/out_fname, 'r') as handle:

            # read the data and determine the fraction of english words
            data = handle.read().replace('\n', ' ')
            words = data.split()
            half = int(len(words)/2)
            N_total = len(words)
            N_english = len([w for w in words if w.lower() in english_words])
            fraction = N_english/(N_total+0.001)

            # remove if smaller than 0.5 (manually selected value found by inspection)
            # french have 0.4-0.5, english >0.55, german < 0.3, bad transcriptions < 0.15
            if fraction < 0.5:
                print(fraction)
                print(words[half:half+40])
                os.system('rm {} {} {}'.format(in_path/fname, mid_path/out_fname, out_path/out_fname))

    except FileNotFoundError:
        os.system('rm {} {} {}'.format(in_path/fname, mid_path/out_fname, out_path/out_fname))


def get_all_chars():

    # all characters
    all_chars = set()
    csets = []

    # loop over all files and extract used characters
    for fname in os.listdir(out_path):
        print(fname)

        with open(out_path/fname, 'r') as handle:
            txt = handle.read()
            these_chars = set(txt)
            all_chars = all_chars.union(these_chars)
            csets.append(these_chars)

    # in how many theses do individual characters appear
    counts = {c:0 for c in all_chars}
    for cset in csets:
        for c in cset:
            counts[c] += 1

    sorted_chars = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # only allow characters used in more than a third of the theses
    N_theses = max(counts.values())
    allowed_chars = [c for c in counts if counts[c] > N_theses/2.]

    return allowed_chars


def convert_to_text():

    english_words = set(nltk.corpus.words.words())
    for i, fname in enumerate(os.listdir(in_path)):
        
        # only look at pdfs
        if not Path(fname).suffix == '.pdf':
            continue
        out_fname = Path(fname).with_suffix('.txt')
    
        print(fname)
    
        # if already done, don't bother
        if (mid_path/out_fname).exists():
            print('{} exists, not converting'.format(out_fname))
        else:
            # turn into text and remove empty lines
            os.system('pdftotext {} {}'.format(in_path/fname, mid_path/out_fname))
    
        # do the postprocessing
        with open(out_path/out_fname, 'w') as handle:
    
            try:
                for j, line in enumerate(generate_lines(mid_path/out_fname)):
                    line = line.strip()
    
                    keep = True
                    keep = keep and has_3_or_more_spaces(line)
                    keep = keep and is_not_toc(line)
                    keep = keep and has_alphabet_majority(line)
    
                    if not keep:
                        continue
                    handle.write(line+'\n')
    
            except FileNotFoundError:
                print('FileNotFoundError caught for {}'.format(fname))
    
        # remove failed transcriptions and non-english pdfs
        remove_failed_transcriptions(fname, out_fname)
        remove_nonenglish_theses(fname, out_fname, english_words)
    
        if i >= 100:
            pass
            #break

def remove_rare_characters(allowed_chars):

    for i, fname in enumerate(os.listdir(in_path)):
        
        print(fname)

    break
    

def main():

    #convert_to_text()
    allowed_chars = get_all_chars()
    print(sorted(allowed_chars))
    remove_rare_characters(allowed_chars)

if __name__ == '__main__':
    main()
