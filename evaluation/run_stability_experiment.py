import numpy as np
import os
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

os.makedirs('results',exist_ok=True)
os.makedirs('figures',exist_ok=True)

cmd = '../build/evaluation/test_random_problems -point_noise 0 -ntrials 10000 -output_path ./results/stability.tab'
os.system(cmd)

matplotlib.rcParams.update({'font.size': 10})
#matplotlib.rcParams.update({'figure.figsize': [3.5,2]})
matplotlib.rcParams.update({'figure.figsize': [3.5,2.5]})

path = 'results/stability.tab'
#print(path)
res = np.loadtxt(path)
good = np.isinf(res)
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

action = get(res,0)
poly = get(res,3)
nister = get(res,6)
stew = get(res,9)

"""
print('frob medians: action, poly, fivept')
print(np.median(action[0]),\
      np.median(poly[0]),\
      np.median(fivept[0]))
print('rot means: 3q3, koeser, ippe, p3p')
print(np.mean(acp1p_ang),\
      np.mean(koeser_ang),\
      np.mean(ippe_ang),\
      np.mean(p3p_ang))

print('pos medians: 3q3, koeser, ippe, p3p')
print(np.median(acp1p_pos),\
      np.median(koeser_pos),\
      np.median(ippe_pos),\
      np.median(p3p_pos))
print('pos means: 3q3, koeser, ippe, p3p')
print(np.mean(acp1p_pos),\
      np.mean(koeser_pos),\
      np.mean(ippe_pos),\
      np.mean(p3p_pos))
"""

def make_pdf_plot(data,bins,ax):
    data = data[np.logical_not(np.isnan(data))]
    data = data[np.logical_not(np.isinf(data))]
    density = gaussian_kde(data)
    ax.plot(bins,density(bins))

frob_bins = np.linspace(-16,0,1000)

names = ['Action Matrix','Polynomial','Nister','Stewenius']

fig,ax = plt.subplots()
make_pdf_plot(np.log10(action[0]),frob_bins,ax)
make_pdf_plot(np.log10(poly[0]),frob_bins,ax)
make_pdf_plot(np.log10(nister[0]),frob_bins,ax)
make_pdf_plot(np.log10(stew[0]),frob_bins,ax)
ax.set_xlabel('log(error)')
ax.set_ylabel('density')
ax.legend(names)
ax.set_title('Log Frobenius norm')
plt.tight_layout()
fig.savefig('figures/stability_frob.eps')
fig.savefig('figures/stability_frob.png')

"""
fig,ax = plt.subplots()
make_pdf_plot(np.log10(p3p_pos),pos_bins,ax)
make_pdf_plot(np.log10(acp1p_pos),pos_bins,ax)
make_pdf_plot(np.log10(ippe_pos),pos_bins,ax)
make_pdf_plot(np.log10(koeser_pos),ang_bins,ax)
ax.legend(names)
ax.set_xlabel('log(error)')
ax.set_ylabel('density')
ax.set_title('Log position error')
plt.tight_layout()
fig.savefig('stability_pos.eps')
fig.savefig('stability_pos.png')
"""

