Requirements
We test the code in the following environments, other versions may also be compatible:

CUDA 11.1
Python 3.7
Pytorch 1.7.1

### Setup
First, clone the repository locally and create conda enviroment .
```
git clone https://github.com/Ly-Chen111/MaSTAN.git
conda create -n mastan python=3.7
conda activate mastan
```

Then, install Pytorch 1.7.1.
```
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
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