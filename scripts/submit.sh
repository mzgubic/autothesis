# change n_cores
for cell in RNN GRU LSTM
do
    for hsize in 8 16 32 64 128 256
    do
        for maxlen in 10 20 50 100
        do
            py train.py --force --condor --n-epochs 1 --cell $cell --hidden-size $hsize --max-len $maxlen
        done
    done
done
