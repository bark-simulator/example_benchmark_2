#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/tools/python/venv/lib/python3.7/site-packages/torch/lib/ 