# change n_cores
for cell in GRU LSTM
do
    for hsize in 32 64 128
    do
        for maxlen in 20
        do
            py train.py --force --condor --n-epochs 10 --cell $cell --hidden-size $hsize --max-len $maxlen
        done
    done
done
