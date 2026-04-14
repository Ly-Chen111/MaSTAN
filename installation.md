### Requirements
We test the code in the following environments, other versions may also be compatible:

CUDA 11.1
Python 3.8
Pytorch 1.8.1

### Setup
First, clone the repository locally and create conda enviroment .
```
git clone https://github.com/Ly-Chen111/MaSTAN.git
conda create -n mastan python=3.8 -y
conda activate mastan
```

Then, install Pytorch 1.8.1.
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install requirements.txt 
```
pip install -r requirements.txt 
```

Finally, compile CUDA operators.
```
cd models/ops
python setup.py build install
cd ../..
```

NOTE: If you have problem installing MultiScaleDeformableAttention_update, please:
1. delete the dictionary in your conda envs -> your-conda-path/envs/mastan/lib/python3.7/site-packages/MultiScaleDeformableAttention-1.0-py3.7-linux-x86_64.egg.
2. delete the dist,build and MultiScaleDeformableAttention_update.egg-info in models/ops.
3. export your CUDA_HOME like: export CUDA_HOME=/usr/local/cuda.
4. recompile.
