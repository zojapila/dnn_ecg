#!/bin/bash

conda install python=3.11
conda install scikit-learn
pip install tensorflow
conda install wfdb=4.1.1
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install anaconda::scipy