# change n_cores
for cell in RNN GRU LSTM
do
    py train.py --force --n-epochs 10 --cell $cell --condor
done
