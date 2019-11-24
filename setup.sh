#!/bin/bash
if [ $USER == "zgubic" ] && [[ $HOSTNAME == "pplxint"* ]]; then
    export SRC="/home/zgubic/thesis/autothesis"
    export DATA="/data/atlassmallfiles/users/zgubic/thesis"
elif [ $USER == "zgubic" ] && [[ $HOSTNAME == "pposx"* ]]; then
    export SRC="/Users/zgubic/Projects/autothesis"
    export DATA="/Users/zgubic/Projects/autothesis"
fi

# for the moment, need to use the most recent nightly build
source ${SRC}/bin/activate
export PYTHONPATH="$PYTHONPATH:${SRC}/autothesis"

