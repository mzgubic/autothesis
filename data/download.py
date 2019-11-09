import os
from pathlib import Path

in_path = Path('extracted_https')
out_path = utils.data_path / 'pdfs'

for fname in os.listdir(in_path):
    print(fname)
    with open(in_path/fname) as handle:
        for line in handle:
            print(line)
            http = line.strip()
            pdfname = http.split('/')[-1]

            already_have = os.path.exists(out_path/pdfname)

            if already_have:
                print('already have {}'.format(pdfname))
                continue
            else:
                command = 'wget {}'.format(http)
                os.system(command)
                os.system('mv {} {}'.format(pdfname, out_path/pdfname))
