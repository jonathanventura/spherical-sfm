import numpy as np
import os
from tqdm import tqdm

cmd = '../build/evaluation/test_random_problems -point_noise 0 -ntrials 10000 -timings_path results/timings.txt'
os.system(cmd)

a = np.loadtxt('results/timings.txt',usecols=(0,1,2,3),delimiter='\t')
#print(['Null','Elim','3Q3','P3P'])
print(['Action','Poly','Nister','Stewenius'])
print(np.mean(a,axis=0)*1000)
