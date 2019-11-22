# change n_cores
for cell in RNN GRU LSTM
do
    for hsize in 64 128 256
    do
        py train.py --force --condor --n-epochs 1 --cell $cell --hidden-size $hsize
    done
done
