from time import time
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from SpeedUp import LocalNMF, GetCenters

plt.rc('figure', facecolor='white', dpi=90, frameon=False)
plt.rc('font', size=44, **{'family': 'sans-serif', 'sans-serif': ['Computer Modern']})
plt.rc('lines', lw=2)
plt.rc('text', usetex=True)
plt.rc('legend', **{'fontsize': 36})
plt.rc('axes', linewidth=2)
plt.rc('xtick.major', size=10, width=1.5)
plt.rc('ytick.major', size=10, width=1.5)

# colors for colorblind from
# http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/
col = ["#D55E00", "#E69F00", "#F0E442", "#009E73", "#56B4E9", "#0072B2", "#CC79A7", "#999999"]
# vermillon, orange, yellow, green, cyan, blue, purple, grey


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# Fetch Data, take only 100x100 patch to not have to wait minutes
cl = 200 if len(argv) == 1 else int(argv[1])

sig = (4, 4)
lam = 40
data = np.asarray([np.load('../zebrafish/ROI_zebrafish/data/1/nparrays/TM0%04d_%d-%d_350-550_15.npy' % (t, cl, cl + 200))
                   for t in range(3000)])[:, : 100, : 100]
x = np.load('x%d.npy' % cl)[:, : 100, : 100]  # x is stored result from grouplasso
pic_x = np.percentile(x, 95, 0)
cent = GetCenters(pic_x)

MSE_array, shapes, activity, boxes = LocalNMF(data, (np.array(cent)[:-1]).T,
                                              sig, verbose=True, iters=50, iters0=[100], mbs=[30])
np.savez('%d/result.npz' % cl, MSE_array=MSE_array, shapes=shapes, activity=activity, boxes=boxes)

MSE_array0, shapes0, activity0, boxes0 = LocalNMF(data, (np.array(cent)[:-1]).T,
                                                  sig, verbose=True, iters=100, iters0=[0])
np.savez('%d/resultNoSS.npz' % cl, MSE_array=MSE_array0,
         shapes=shapes0, activity=activity0, boxes=boxes0)

# plt.figure()
# plt.plot(activity[0] / activity[0].max())
# plt.plot(1.1 + activity[5] / activity[5].max())
# plt.plot(2.2 + activity[11] / activity[11].max())
# plt.ylim(0, 3.2)
# plt.xticks([0, 1200, 2400], [0, 10, 20])
# plt.yticks([])
# plt.xlabel('Time [min]')
# plt.ylabel('Activity [a.u.]')
# simpleaxis(plt.gca())
# plt.subplots_adjust(.09, .22, .99, .99)
# plt.savefig('%d/seriesNMF.pdf' % cl, dpi=600, transparent=True)

from operator import itemgetter
MSE_array0, shapes0, activity0, boxes0 = itemgetter(
    'MSE_array', 'shapes', 'activity', 'boxes')(np.load('%d/resultNoSS.npz' % cl))
MSE_array, shapes, activity, boxes = itemgetter(
    'MSE_array', 'shapes', 'activity', 'boxes')(np.load('%d/result.npz' % cl))

centers = (np.array(cent)[:-1]).T
idx = [2, 6, 16]

plt.figure()  # figsize=(8, 8),frameon = False)
plt.imshow(-data.max(0).T, 'Greys')
for b in boxes:
    plt.gca().add_patch(plt.Rectangle(b[:, 0], b[0, 1] - b[0, 0] - 1, b[1, 1] - b[1, 0] - 1,
                                      linestyle='dotted', lw=1.5, fill=False, ec=np.random.rand(3, 1)))
for i, k in enumerate(idx):
    plt.gca().add_patch(plt.Rectangle(boxes[k][:, 0],
                                      boxes[k][0, 1] - boxes[k][0, 0] - 1,
                                      boxes[k][1, 1] - boxes[k][1, 0] - 1,
                                      lw=1.5, fill=False, ec=col[i]))
    plt.scatter(*centers[k], s=60, marker='x', lw=3, c=col[i], zorder=10)
plt.axis('off')
# plt.gca().set_axis_off()
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig('%d/patch' % cl, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)


plt.figure()
for i, k in enumerate(idx):
    plt.plot(1.1 * i + activity0[k] / activity0[k, 1200:2400].max(),
             lw=3, c='k')  # , alpha=.5)
    plt.plot(1.1 * i + activity[k] / activity[k, 1200:2400].max(), lw=1, c=col[i])  # , alpha=.5)
plt.xlim(1200, 2400)
plt.ylim(0, 3.2)
plt.xticks([1200, 2400], [10, 20])
plt.yticks([])
plt.xlabel('Time [min]', labelpad=-15)
plt.ylabel('Activity [a.u.]')
simpleaxis(plt.gca())
plt.subplots_adjust(.09, .17, .96, .99)
plt.savefig('%d/seriesNMF.pdf' % cl, dpi=600, transparent=True)


fig = plt.figure(figsize=(9, 6))
for i, k in enumerate(idx):
    ax = fig.add_axes([i / 3., .5, 1 / 3., .5])
    ax.imshow(-shapes[k].reshape(100, 100)[map(lambda a: slice(*a), boxes[k])].T, cmap='Greys')
    ax.scatter([1.5], [1.5], s=300, marker='x', lw=7, c=col[i])
    ax.axis('off')
    ax = fig.add_axes([i / 3., 0, 1 / 3., .5])
    ax.imshow(-shapes0[k].reshape(100, 100)[map(lambda a: slice(*a), boxes0[k])].T, cmap='Greys')
    ax.scatter([1.5], [1.5], s=300, marker='x', lw=7, c='k')
    ax.axis('off')
plt.savefig('%d/shapesNMF.pdf' % cl, dpi=600, transparent=True)


MSE_arrayQ, shapesQ, activityQ, boxesQ = LocalNMF(data, (np.array(cent)[:-1]).T,
                                              sig, verbose=True, iters=0, iters0=[80], mbs=[30])

plt.figure()
for i, k in enumerate(idx):
    plt.plot(1.1 * i + activity0[k] / activity0[k, 1200:2400].max(),
             lw=3, c='k')  # , alpha=.5)
    plt.plot(1.1 * i + activityQ[k] / activityQ[k, 1200:2400].max(), lw=1, c=col[i])  # , alpha=.5)
plt.xlim(1200, 2400)
plt.ylim(0, 3.2)
plt.xticks([1200, 2400], [10, 20])
plt.yticks([])
plt.xlabel('Time [min]', labelpad=-15)
plt.ylabel('Activity [a.u.]')
simpleaxis(plt.gca())
plt.subplots_adjust(.09, .17, .96, .99)
plt.savefig('%d/seriesQNMF.pdf' % cl, dpi=600, transparent=True)

fig = plt.figure(figsize=(9, 6))
for i, k in enumerate(idx):
    ax = fig.add_axes([i / 3., .5, 1 / 3., .5])
    ax.imshow(-shapesQ[k].reshape(100, 100)[map(lambda a: slice(*a), boxes[k])].T, cmap='Greys')
    ax.scatter([1.5], [1.5], s=300, marker='x', lw=7, c=col[i])
    ax.axis('off')
    ax = fig.add_axes([i / 3., 0, 1 / 3., .5])
    ax.imshow(-shapes0[k].reshape(100, 100)[map(lambda a: slice(*a), boxes0[k])].T, cmap='Greys')
    ax.scatter([1.5], [1.5], s=300, marker='x', lw=7, c='k')
    ax.axis('off')
plt.savefig('%d/shapesQNMF.pdf' % cl, dpi=600, transparent=True)



plt.figure()
for i, k in enumerate(idx):
    plt.plot(1.1 * i + activity0[k] / activity0[k, 1200:2400].max(),
             lw=2, c='k')  # , alpha=.5)
    tmp = data[np.ix_(range(3000), int(centers[k, 0]) + np.arange(-1, 2),
                             int(centers[k, 1]) + np.arange(-1, 2))]
    activityMean = (tmp - np.percentile(tmp, 20, 0)).mean(1).mean(1)
    plt.plot(1.1 * i + activityMean / activityMean[1200:2400].max(), lw=1, c=col[i])  # , alpha=.5)
plt.xlim(1200, 2400)
plt.ylim(0, 3.2)
plt.xticks([1200, 2400], [10, 20])
plt.yticks([])
plt.xlabel('Time [min]', labelpad=-15)
plt.ylabel('Activity [a.u.]')
simpleaxis(plt.gca())
plt.subplots_adjust(.09, .17, .96, .99)
plt.savefig('%d/seriesMean.pdf' % cl, dpi=600, transparent=True)



# plt.figure()
# for i,k in enumerate([0,5,11]):
#     plt.plot(1.1*i + activity[k] / activity[k].max(), c=col[i], alpha=.5)
#     plt.plot(1.1*i + activity0[k] / np.median(activity0[k])*np.median(activity[k])/ activity[k].max(), c=col[i+3], alpha=.5)
# plt.ylim(0, 3.2)
# plt.xticks([0, 1200, 2400], [0, 10, 20])
# plt.yticks([])
# plt.xlabel('Time [min]')
# plt.ylabel('Activity [a.u.]')
# simpleaxis(plt.gca())
# plt.subplots_adjust(.09, .22, .99, .99)

#
#
#
#
#
#

# iterls = np.outer([10, 20, 40, 60, 80], np.ones(2, dtype=int)) / 2
iterls = [0, 20, 40, 60, 80, 100]
# iterls = [80]

# with downsampling shapes
try:
    MSE_array = np.load('%d/MSE.npy' % cl)
except:
    MSE_array = [LocalNMF(data, (np.array(cent)[:-1]).T, sig, verbose=True, iters=50, iters0=[i], mbs=[30])[0]
                 for i in iterls]
    np.save('%d/MSE' % cl, MSE_array)

plt.figure()
for i, m in enumerate(MSE_array):
    plt.plot(np.array(m)[1:, 0], np.array(m)[1:, 1] / data.size,
             label=str(iterls[i]).rjust(3), c=col[i])
plt.xticks([0, 5, 10], [0, 5, 10])
plt.yticks([80, 81], [80, 81])
plt.xlim(0, 10.5)
plt.ylim(80, 81.08)
lg = plt.legend(title='\# subset iterations', fontsize=30, ncol=2, columnspacing=1,
                bbox_to_anchor=(1.06, 1.07), handlelength=2, handletextpad=.2)
for i, t in enumerate(lg.get_texts()):  # right align
    t.set_ha('right')
    t.set_position((25 + i / 3 * 10, 0))
lg.draw_frame(False)
lg.get_title().set_fontsize('32')
plt.xlabel('Wall time [s]')
plt.ylabel('MSE', labelpad=0)
simpleaxis(plt.gca())
plt.subplots_adjust(.15, .22, .99, .99)
plt.savefig('%d/MSE.pdf' % cl, dpi=600, transparent=True)


# without downsampling shapes
try:
    MSE_array2 = np.load('%d/MSEnoSSS.npy' % cl)
except:
    MSE_array2 = [LocalNMF(data, (np.array(cent)[:-1]).T, sig, verbose=True, iters=50,
                           iters0=[i], mbs=[30], ds=1)[0]
                  for i in iterls]
    np.save('%d/MSEnoSSS' % cl, MSE_array2)

plt.figure()
for i, m in enumerate(MSE_array2):
    if i == 0:
        m = MSE_array[0]
    plt.plot(np.array(m)[:, 0], np.array(m)[:, 1] / data.size,
             label=iterls[i], c=col[i])
plt.xticks([0, 5, 10], [0, 5, 10])
plt.yticks([80, 81], [80, 81])
plt.xlim(0, 10.5)
plt.ylim(80, 81.08)
lg = plt.legend(title='\# subset iterations', fontsize=30, ncol=2, columnspacing=1,
                bbox_to_anchor=(1.06, 1.07), handlelength=2, handletextpad=.2)
for i, t in enumerate(lg.get_texts()):  # right align
    t.set_ha('right')
    t.set_position((25 + i / 3 * 10, 0))
lg.draw_frame(False)
lg.get_title().set_fontsize('32')
plt.xlabel('Wall time [s]')
plt.ylabel('MSE', labelpad=0)
simpleaxis(plt.gca())
plt.subplots_adjust(.15, .22, .99, .99)
plt.savefig('%d/MSEnoSSS.pdf' % cl, dpi=600, transparent=True)


# plot with and without downsampling shapes
plt.figure()
for i, m in enumerate(MSE_array2):
    if i == 0:
        m = MSE_array[0]
    plt.plot(np.array(m)[:, 0], np.array(m)[:, 1] / data.size,
             label=iterls[i], c=col[i])
plt.xticks([0, 5, 10], [0, 5, 10])
plt.yticks([80, 81], [80, 81])
plt.xlim(0, 10.5)
plt.ylim(80, 81.08)
lg = plt.legend(title='\# subset iterations', fontsize=30, ncol=2, columnspacing=1,
                bbox_to_anchor=(1.06, 1.07), handlelength=2, handletextpad=.2)
for i, t in enumerate(lg.get_texts()):  # right align
    t.set_ha('right')
    t.set_position((25 + i / 3 * 10, 0))
lg.draw_frame(False)
lg.get_title().set_fontsize('32')
plt.xlabel('Wall time [s]')
plt.ylabel('MSE', labelpad=0)
for i, m in enumerate(MSE_array):
    plt.plot(np.array(m)[1:, 0], np.array(m)[1:, 1] / data.size,
             '--', label=iterls[i], c=col[i])
simpleaxis(plt.gca())
plt.subplots_adjust(.15, .22, .99, .99)
plt.savefig('%d/MSE+noSSS.pdf' % cl, dpi=600, transparent=True)


# outer loop over neurons
try:
    MSE_array3 = np.load('%d/MSElon.npy' % cl)
except:
    from SpeedUpOld import LocalNMF as LocalNMF3
    MSE_array3 = LocalNMF3(data, (np.array(cent)[:-1]).T, sig,
                           verbose=True, iters=50, iters0=0)[0]
    np.save('%d/MSElon' % cl, MSE_array3)

plt.figure()
plt.plot(np.array(MSE_array2[0])[:, 0], np.array(MSE_array2[0])[:, 1] / data.size,
         label='inner loop', c=col[0])
plt.plot(np.array(MSE_array3)[:, 0], np.array(MSE_array3)[:, 1] / data.size,
         label='outer loop', c='k')
plt.xticks([0, 20, 40, 60], [0, 20, 40, 60])
plt.yticks([80, 82, 84], [80, 82, 84])
plt.xlim(0, 63)
plt.ylim(80, 84.3)
lg = plt.legend(title='iterate over neurons', fontsize=30,
                bbox_to_anchor=(1.04, 1.05), handlelength=2, handletextpad=.2)
lg.draw_frame(False)
lg.get_title().set_fontsize('32')
plt.xlabel('Wall time [s]')
plt.ylabel('MSE', labelpad=0)
simpleaxis(plt.gca())
plt.subplots_adjust(.15, .22, .99, .99)
plt.savefig('%d/MSElon.pdf' % cl, dpi=600, transparent=True)

#
#
#
#
#
# OVERLAP
#
#
#

from scipy.ndimage.filters import gaussian_filter
centers = np.array([[16, 13], [16, 20]])
dataOverlapTrue1 = np.zeros((3000, 33, 33))
dataOverlapTrue2 = np.zeros((3000, 33, 33))
centers = np.array([[16, 16], [16, 23]])
dataOverlapTrue1 = np.zeros((3000, 33, 40))
dataOverlapTrue2 = np.zeros((3000, 33, 40))
dataOverlapTrue1[:, centers[0, 0], centers[0, 1]] = activity[0] * shapes[0].max()
dataOverlapTrue2[:, centers[1, 0], centers[1, 1]] = activity[5] * shapes[5].max()
dataOverlapTrue1 = gaussian_filter(dataOverlapTrue1, (0, 5, 5))
dataOverlapTrue2 = gaussian_filter(dataOverlapTrue2, (0, 5, 5))
dataOverlap = 1 + dataOverlapTrue1 + dataOverlapTrue2 + .25 * \
    np.random.randn(*dataOverlapTrue1.shape)
dataOverlap *= (dataOverlap > 0)

plt.figure()
plt.imshow(-np.max(dataOverlap, 0), cmap='Greys')
# import matplotlib.patches as mpatches
plt.gca().add_patch(plt.Circle(centers[0, ::-1], radius=5, color=col[4], lw=4, fill=False))
plt.gca().add_patch(plt.Circle(centers[1, ::-1], radius=5, color=col[1], lw=4, fill=False))
plt.axis('off')
plt.subplots_adjust(0, 0, 1, 1)
plt.savefig('%d/seriesOverlap.pdf' % cl, dpi=600, transparent=True)


def foo(d1, d2):
    fig1, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(d1, c=col[1])  # , clip_on=False)
    ax[1].plot(d2, c=col[4],)  # clip_on=False)
    for i in [0, 1]:
        simpleaxis(ax[i])
        ax[i].set_ylim(0, 1.5)
        ax[i].set_xticks([0, 1200, 2400])
        ax[i].set_yticks([0, 1])
        ax[i].set_yticklabels([0, 1], [0, 1])
    ax[0].set_xticklabels([])
    plt.xticks([0, 1200, 2400], [0, 10, 20])
    ax[1].set_xlabel("Time [min]")
    ax[1].set_ylabel("Activity [a.u.]", y=1)
    plt.subplots_adjust(.14, .22, .99, .99)

foo(dataOverlapTrue2[:, centers[1, 0], centers[1, 1]],
    dataOverlapTrue1[:, centers[0, 0], centers[0, 1]])
plt.savefig('%d/seriesOverlapTruth.pdf' % cl, dpi=600, transparent=True)


from sklearn.decomposition import FastICA, NMF
# ica = FastICA(n_components=2)
# Sica = ica.fit_transform(dataOverlap.reshape(3000, -1))  # Reconstruct signals
# Aica = ica.mixing_  # Get estimated mixing matrix

nmf = NMF(n_components=3)
Snmf = nmf.fit_transform(dataOverlap.reshape(3000, -1))  # Reconstruct signals

foo(nmf.components_[2].max() * Snmf[:, 2], nmf.components_[1].max() * Snmf[:, 1])
plt.savefig('%d/seriesOverlapNMF.pdf' % cl, dpi=600, transparent=True)


activityMean = np.ones((2, 3000))
for ll in range(2):
    tmp = dataOverlap[np.ix_(range(3000), int(centers[ll, 0]) + np.arange(-1, 2),
                             int(centers[ll, 1]) + np.arange(-1, 2))]
    activityMean[ll] = (tmp - np.percentile(tmp, 20, 0)).mean(1).mean(1)
foo(activityMean[1], activityMean[0])
plt.savefig('%d/seriesOverlapMean.pdf' % cl, dpi=600, transparent=True)
