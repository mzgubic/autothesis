# autothesis

Sit back, relax, and let machines write your thesis.

## Set up the code and the environment
```
git clone git@github.com:mzgubic/autothesis.git
cd autothesis
source setup.sh
```

## Get and clean training data

Download raw html of webpages with links to the pdfs from:  
https://cds.cern.ch/collection/ATLAS%20Theses?ln=en  
and put them in the raw_html folder.

Skim thelines with links to the pdfs:
```
. skim.sh
```

Clean up and extract the links
```
python extract_https.py
```

Download the pdf files
```
python download.py
```

And then convert to text, and do dome basic cleaning:
- remove short lines (mostly text from figures)
- remove table of content lines
- remove non english documents
```
python pdf2txt.py
```




