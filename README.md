# autothesis

Sit back, relax, and let machines write your thesis.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mzgubic/autothesis/master?filepath=notebooks%2FGenerateCharacterTokens.ipynb)

## Set up the code and the environment
```
git clone git@github.com:mzgubic/autothesis.git
python3 -m venv autothesis
cd autothesis
pip install -r requirements.txt
source setup.sh
```

## Download training data

Download raw html of webpages with links to the pdfs from:  
https://cds.cern.ch/collection/ATLAS%20Theses?ln=en  
and put them in the raw_html folder.

Skim the lines with links to the pdfs:
```
cd ${SRC}/data
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

## Clean and tokenise

And then convert to text, and do some basic cleaning:
- remove short lines (mostly text from figures)
- remove table of content lines
- remove non english documents
```
cd ${SRC}/scripts
python pdf2txt.py
```




