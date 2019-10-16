import os
from pathlib import Path

in_path = Path('pdfs/')
out_path = Path('txts')


def generate_lines(path):
    
    with open(path, 'r') as handle:
        for line in handle:
            yield line

def has_3_or_more_spaces(line):
    if len(line.split()) > 3:
        return True
    else:
        return False

# convert to txts
for fname in os.listdir(in_path):
    
    # only look at pdfs
    if not Path(fname).suffix == '.pdf':
        continue
    out_fname = Path(fname).with_suffix('.txt')

    print(fname)
    print(out_fname)

    # if already done, don't bother
    if (out_path/out_fname).exists():
        print('{} exists'.format(out_fname))
        continue

    # turn into text and remove empty lines
    os.system('pdftotext {} tmp.txt'.format(in_path/fname))

    with open(out_path/out_fname, 'w') as handle:
        for i, line in enumerate(generate_lines('tmp.txt')):
            line = line.strip()

            keep = True
            keep = keep and has_3_or_more_spaces(line)

            if keep:
                pass
                handle.write(line+'\n')

os.system('rm tmp.txt')

    


        



# clean up the txts
#    with open(in_path/fname) as handle:
#        for i, line in enumerate(handle):
#            print(line)
#            if i > 10:
#                break
