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

os.makedirs('../figures',exist_ok=True)

for mode in ['inward','outward']:

    names = ['Spherical (GB)','Spherical (Poly)','Nister','Stewenius']
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

    data = np.loadtxt(f'../results/{mode}_ransac.tab')
    for j,name in enumerate(names):
        res = get(data,3*j)
        frob[name] = res[0]
        rot[name] = res[1]*180/np.pi
        trans[name] = res[2]*180/np.pi

    frob_meds = {}
    rot_meds = {}
    trans_meds = {}

    for name in names:
        frob_meds[name] = np.median(frob[name])
        rot_meds[name] = np.median(rot[name])
        trans_meds[name] = np.median(trans[name])
    
    print(trans_meds)
    
    
    plt.clf()
    plt.boxplot(trans.values(),labels=trans.keys(),whis=(0,100))
    #plt.xticks(range(1,12),range(0,11))
    #plt.ylim(0,30)
    plt.ylabel('Translation error (deg)')
    plt.savefig(f'../figures/{mode}_ransac_trans.png')

    plt.clf()
    plt.boxplot(rot.values(),labels=rot.keys(),whis=(0,100))
    plt.ylabel('Rotation error (deg)')
    plt.savefig(f'../figures/{mode}_ransac_rot.png')

    plt.clf()
    plt.boxplot(frob.values(),labels=frob.keys(),whis=(0,100))
    plt.ylabel('Frobenius error')
    plt.savefig(f'../figures/{mode}_ransac_frob.png')

    """

    plt.clf()
    for name in names:
        plt.plot(obs_noise,rot_meds[name])
    plt.legend(names)
    plt.savefig(f'figures/rot_{mode}_noise.png')

    plt.clf()
    for name in names:
        plt.plot(obs_noise,trans_meds[name])
    plt.legend(names)
    plt.savefig(f'figures/trans_{mode}_noise.png')

    for name in names:
        plt.clf()
        plt.boxplot(frob[name],whis=(0,100))
        plt.xticks(range(1,12),range(0,11))
        plt.ylim(0,0.5)
        plt.savefig(f'figures/frob_{mode}_noise_{name}.png')

        plt.clf()
        plt.boxplot(rot[name],whis=(0,100))
        plt.xticks(range(1,12),range(0,11))
        plt.ylim(0,5)
        plt.savefig(f'figures/rot_{mode}_noise_{name}.png')

        plt.clf()
        plt.boxplot(trans[name],whis=(0,100))
        plt.xticks(range(1,12),range(0,11))
        plt.ylim(0,50)
        plt.savefig(f'figures/trans_{mode}_noise_{name}.png')
    """

