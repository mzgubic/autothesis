import os
from pathlib import Path


in_path = Path('skimmed_html')
out_path = Path('extracted_https')

for fname in os.listdir(in_path):
    print(fname)
    these_https = []
    with open(in_path/fname) as handle:
        for line in handle:
            components = line.split('a> - <a')
            https = [c.split('"')[1] for c in components if 'cds.cern.ch' in c and 'pdf' in c]
            try:
                these_https.append(https[-1])
            except IndexError:
                print('whoops, no theses, skipping')

    with open(out_path/fname, 'w') as handle:
        for http in these_https:
            print(http)
            handle.write(http+'\n')
        


