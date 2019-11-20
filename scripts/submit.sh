# change n_cores
for i in {1..16}
do
    py train.py --force --n-steps 100000 --cell RNN --condor --n-cores $i
done
