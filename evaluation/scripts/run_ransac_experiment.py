import subprocess
import numpy as np
import os
from tqdm import tqdm

obs_noise = 1
ntrials = 100
rotation = 2

os.makedirs('../results',exist_ok=True)

cmd = f'../../build/evaluation/test_ransac -point_noise {obs_noise} -ntrials {ntrials} -output_path ../results/inward_ransac.tab -rotation {rotation} -inward '
os.system(cmd)

cmd = f'../../build/evaluation/test_ransac -point_noise {obs_noise} -ntrials {ntrials} -output_path ../results/outward_ransac.tab -rotation {rotation}'
os.system(cmd)

