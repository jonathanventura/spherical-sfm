import os
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

df = pd.read_csv('../build/results.csv')

df_nister = df[df['method'] == 'nister']
df_stewenius = df[df['method'] == 'stewenius']
df_threept = df[df['method'] == 'threept']

# angles = np.arange(1,101)/10
noises = sorted(df['noise'].unique())

nister_error = df_nister[['noise','E_err']].groupby(['noise']).mean().values
stewenius_error = df_stewenius[['noise','E_err']].groupby(['noise']).mean().values
threept_error = df_threept[['noise','E_err']].groupby(['noise']).mean().values
plt.plot(noises,nister_error)
plt.plot(noises,stewenius_error)
plt.plot(noises,threept_error)
plt.xlabel('Noise std. dev. [px]')
plt.ylabel('E error')
plt.legend(['Nister','Stewenius','Spherical'])
plt.savefig('E_error_with_noise.png')

plt.clf()
nister_error = df_nister[['noise','t_err']].groupby(['noise']).mean().values
stewenius_error = df_stewenius[['noise','t_err']].groupby(['noise']).mean().values
threept_error = df_threept[['noise','t_err']].groupby(['noise']).mean().values
plt.plot(noises,nister_error*180/np.pi)
plt.plot(noises,stewenius_error*180/np.pi)
plt.plot(noises,threept_error*180/np.pi)
plt.xlabel('Noise std. dev. [px]')
plt.ylabel('Avg. trans. error [deg]')
plt.legend(['Nistér','Stewénius','Spherical'])
plt.savefig('t_error_with_noise.png')
