# gym-ctr-generic

## Step to get environment setup.

1. Clone this repo and switch to system_agnostic branch
- ```git clone https://github.com/keshaviyengar/gym-ctr-generic.git```
- ```git checkout system_agnostic```
2. Clone the stable-baselines repo and switch to hill-a-master branch
- ```git clone https://github.com/keshaviyengar/stable-baselines.git```
- ```git checkout hill-a-master```
3. Clone the rl-baselines-zoo repo and switch to araffin-master
- ```git clone https://github.com/keshaviyengar/rl-baselines-zoo.git```
- ```git checkout araffin-master```
4. Create a python virtual environment
- ```python3 -m venv venv/```
- ``` source venv/bin/activate```
- ```python -m pip install --upgrade pip```
6. Install stable-baselines and gym-ctr-generic locally
- ```pip install -e stable-baselines/```
- ```pip install -e gym-ctr-generic/```
7. Test by running training example
- ``` python train.py --env "CTR-Generic-Reach-v0" --algo "her" --gym-packages "ctr_generic_envs" --log-interval 2 --experiment "free_rotation/tro_free_0"```


## Common errors
```AttributeError: module 'contextlib' has no attribute 'nullcontext'```
Install older version (0.15.7) of gym or yse python3.7.
```ModuleNotFoundError: No module named 'yaml'```
If a module is missing, install with pip
DDPG needs MPI
```pip install mpi4py```
