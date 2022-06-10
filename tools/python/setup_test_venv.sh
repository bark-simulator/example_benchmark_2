#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/venv
virtualenv --system-site-packages -p python3.7 $DIR/venv
source tools/python/venv/bin/activate && pip install -r tools/gluster/installers/requirements.txt
pip install torch==1.9.0+cpu torchvision -f https://download.pytorch.org/whl/torch_stable.html