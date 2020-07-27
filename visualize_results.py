"""
Result visualization
"""
# Import packages
import numpy as np
import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns

import os

# Folder containing benchmarking results
filepath = '/mnt/d/GitHub/connectfour_agent/agents/benchmarking/results/'

# Simulated games per second ===========================================================================================
# Find the files with _nns.csv termination
nssf = [os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('_nss.csv')]

# Extract data from the files
data = np.array([np.loadtxt(file, delimiter=",") for file in nssf])
implementation_tag = [re.sub('.*/(.*)_benchmarks_nss.csv', '\\1', file) for file in nssf]

# Get limits of the distributions -> used later for uniform binning of both data sets
lims = np.stack([data.min(axis=1), data.max(axis=1)]).T

# Visualize distributions
[sns.distplot(data[i], bins=np.arange(lims[i, 0], lims[i, 1], 5),
              label=tag+r' $\bar{x}$='+f'{np.round(data[i].mean())}') for i, tag in enumerate(implementation_tag)]
plt.gca().set(xlabel='games played per second', ylabel='pdf()',
              title=f'MCTS implementations\nnumber of games simulated')
plt.legend()
plt.show()

# C parameter tuning ==================================================================================================
# Find files with _wr.csv termination
wrf = [os.path.join(filepath, file) for file in os.listdir(filepath) if file.endswith('_wr.csv')]

# Get win rate data
data = np.array([np.loadtxt(file, delimiter=",").mean(axis=0) for file in wrf])
cvals = np.array([re.sub('.*benchmarks_c(.*)_wr\.csv', '\\1', file) for file in wrf])

# Separate in C values and annealing schedules
cfilt = np.array([False if c.find('exp') == 0 else True for c in cvals])

# Only c
c = np.array(cvals[cfilt], dtype=float)
dat = data[cfilt]

# Sort them
sort_idx = np.argsort(c)
c = c[sort_idx]
dat = dat[sort_idx]

# Make data frame for easy bar plots
cvals_col = np.hstack([c, c])
data_col = dat.T.flatten()
tag_col = np.repeat(['P1', 'P2'], len(c))
d = {'c': cvals_col, 'win_rate': data_col, 'player': tag_col}
df = pd.DataFrame(d)

plt.figure(figsize=(10,5))
sns.barplot(x='c', y='win_rate', hue='player', data=df)
plt.title('C parameter tuning\nWin rate against minimax agent')
plt.show()

# Only c
c = np.array(cvals[~cfilt])
dat = data[~cfilt]

# Sort them
sort_idx = np.argsort(c)
c = c[sort_idx]
dat = dat[sort_idx]

# Make data frame for easy bar plots
cvals_col = np.hstack([c, c])
data_col = dat.T.flatten()
tag_col = np.repeat(['P1', 'P2'], len(c))
d = {'c': cvals_col, 'win_rate': data_col, 'player': tag_col}
df = pd.DataFrame(d)

sns.barplot(x='c', y='win_rate', hue='player', data=df)
plt.title('C annealing schedules\nWin rate against minimax agent')
plt.show()

# Plot annealing schedules
t = np.arange(0, 5.0, 0.01)
c0 = 10
c = c0*np.exp(-t)
c1 = c0*np.exp(-t*0.3)
c2 = c0*np.exp(-t*0.7)
c3 = c0*np.exp(-t*1.3)
[plt.plot(t, x) for x in [c1, c2, c, c3]]
plt.gca().set(xlabel='time [s]', ylabel='c param',
              title='C parameter annealing schedule')
plt.legend(['exp_a', 'exp_b', 'exp_c', 'exp_d'])
plt.show()