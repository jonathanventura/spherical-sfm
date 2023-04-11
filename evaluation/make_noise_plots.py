import numpy as np
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'figure.figsize': [7,3.2]})

obs_noise = np.linspace(0,10,11)

names = ['action','poly','nister','stew']
frob = {}
rot = {}
trans = {}
for name in names:
    frob[name] = []
    rot[name] = []
    trans[name] = []

def get(res,i):
    frob = res[:,i]
    rot = res[:,i+1]
    trans = res[:,i+2]
    good_frob = ~np.isinf(frob)
    good_rot = ~np.isinf(rot)
    good_trans = ~np.isinf(trans)
    frob_res = frob[good_frob]
    rot_res = rot[good_rot]
    trans_res = trans[good_trans]
    return frob_res, rot_res, trans_res

for i,noise in enumerate(obs_noise):
    data = np.loadtxt(f'results/noise_{i}.tab')
    for j,name in enumerate(names):
        res = get(data,3*j)
        frob[name].append(res[0])
        rot[name].append(res[1]*180/np.pi)
        trans[name].append(res[2]*180/np.pi)

frob_meds = {}
rot_meds = {}
trans_meds = {}

for name in names:
    frob_meds[name] = np.array([np.median(a) for a in frob[name]])
    rot_meds[name] = np.array([np.median(a) for a in rot[name]])
    trans_meds[name] = np.array([np.median(a) for a in trans[name]])

plt.clf()
for name in names:
    plt.plot(obs_noise,frob_meds[name])
plt.legend(names)
plt.savefig(f'figures/frob_noise.png')

plt.clf()
for name in names:
    plt.plot(obs_noise,rot_meds[name])
plt.legend(names)
plt.savefig(f'figures/rot_noise.png')

plt.clf()
for name in names:
    plt.plot(obs_noise,trans_meds[name])
plt.legend(names)
plt.savefig(f'figures/trans_noise.png')

for name in names:
    plt.clf()
    plt.boxplot(frob[name],whis=(0,100))
    plt.xticks(range(1,12),range(0,11))
    plt.ylim(0,0.5)
    plt.savefig(f'figures/frob_noise_{name}.png')

    plt.clf()
    plt.boxplot(rot[name],whis=(0,100))
    plt.xticks(range(1,12),range(0,11))
    plt.ylim(0,5)
    plt.savefig(f'figures/rot_noise_{name}.png')

    plt.clf()
    plt.boxplot(trans[name],whis=(0,100))
    plt.xticks(range(1,12),range(0,11))
    plt.ylim(0,20)
    plt.savefig(f'figures/trans_noise_{name}.png')

