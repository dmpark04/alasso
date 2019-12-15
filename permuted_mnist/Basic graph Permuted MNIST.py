
# coding: utf-8

import sys, os
sys.path.extend([os.path.expanduser('..')])
from alasso import utils

import matplotlib as mpl
mpl.use('Agg')

import seaborn as sns
sns.set_style("white")

from tqdm import trange, tqdm

import matplotlib.colors as colors
import matplotlib.cm as cmx

import numpy as np
import matplotlib.pyplot as plt
figure = plt.figure
plot = plt.plot
subplot = plt.subplot
xlabel = plt.xlabel
ylabel = plt.ylabel
legend = plt.legend
ylim = plt.ylim
xlim = plt.xlim
gcf = plt.gcf
grid = plt.grid
axhline = plt.axhline
savefig = plt.savefig

import glob
listing = glob.glob('./*.gz')
if len(listing) != 1:
    print("Error: more than one gz file")
    exit()

datafile_name = os.path.basename(listing[0])
file_prefix = os.path.splitext(datafile_name)[0]


diag_vals = dict()
all_evals = dict()

all_evals = utils.load_zipped_pickle(datafile_name)


keys = list(all_evals.keys())
sorted_keys = np.sort(keys)

n_tasks = all_evals[keys[0]].shape[0]


cmap = plt.get_cmap('cool') 
cNorm  = colors.Normalize(vmin=-4, vmax=np.log(np.max(list(all_evals.keys()))))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
print(scalarMap.get_clim())


figure()
f, axs = plt.subplots(1, n_tasks + 1, figsize=(100,100))

for param_index in sorted_keys:
    evals = all_evals[param_index]
    for j in range(n_tasks):
        axs[j].plot(evals[:, j], label="t%d, i%g"%(j, param_index))
        axs[j].set_ylim(0.6, 1.02)
    label = "i=%g"%param_index
    average = evals.mean(1)
    axs[-1].plot(average, label=label)
    
for i, ax in enumerate(axs):
    ax.legend(bbox_to_anchor=(1.0,1.0))
    ax.set_title((['task %d'%j for j in range(n_tasks)] + ['average'])[i])
gcf().tight_layout()

savefig("%s_1.pdf"%(file_prefix))

figure()

for param_index in sorted_keys:
    stuff = []
    for i in range(len(all_evals[param_index])):
        stuff.append(all_evals[param_index][i][:i+1].mean())
    plot(range(1,n_tasks+1), stuff, 'o-', label="i=%g"%param_index)
    
xlabel('Number of tasks')
ylabel('Fraction correct')
legend(loc='best')
ylim(0.6, 0.99)
xlim(0.5, n_tasks + 0.5)
grid('on')

savefig("%s_2.pdf"%(file_prefix))

