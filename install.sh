#!/bin/bash

conda install -y -c conda-forge matplotlib transformers datasets wandb sentence-transformers

conda install -y -c anaconda jupyter

conda install -y pandas

pip install scikit-learn polyfuzz html5lib lxml

pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install git+https://github.com/paperswithcode/axcell.git