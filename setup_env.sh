#!/bin/bash

module load gcc/9.4.0 python/3.8.10 py-virtualenv cuda

virtualenv venv_vessel
source venv_vessel/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install optuna tensorboard matplotlib tqdm pandas scikit-image scipy
