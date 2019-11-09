#!/bin/bash
export SRC="/home/zgubic/thesis/autothesis"
export DATA="/data/atlassmallfiles/users/zgubic/thesis"

# for the moment, need to use the most recent nightly build
source ${SRC}/bin/activate
export PYTHONPATH="$PYTHONPATH:${SRC}/autothesis"

