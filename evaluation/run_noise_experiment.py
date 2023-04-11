import subprocess
import numpy as np
import os
from tqdm import tqdm

obs_noise = np.linspace(0,10,11)

ntrials = 10000

processes = []
for i,obs in tqdm(enumerate(obs_noise)):
    cmd = f'../build/evaluation/test_random_problems -point_noise {obs} -ntrials {ntrials} -output_path results/noise_{i}.tab'
    processes.append(subprocess.Popen(cmd.split(' ')))
for p in processes:
    p.wait()

