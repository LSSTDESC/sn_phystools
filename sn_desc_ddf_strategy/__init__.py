from .version import __version__
import matplotlib.pyplot as plt

filtercolors = dict(zip('ugrizy', ['b', 'c', 'g', 'y', 'r', 'm']))
filtermarkers = dict(zip('ugrizy', ['o', 's', 'P', '*', 'D', 'X']))
filtermarkers = dict(zip('ugrizy', ['o', '*', 's', 'h', '^', 'v']))
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['lines.markersize'] = 15
