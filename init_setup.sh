conda create --prefix ./env python=3.7.6 -y
conda activate ./env
pip3 install torch torchvision torchaudio  # torch for cuda 10.2
pip3 install -r requirements.txt