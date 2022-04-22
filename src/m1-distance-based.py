
from opt import actions_dict
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
from tools import *
from sklearn.neighbors import NearestCentroid
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

actions = actions_dict[3]
colors =['red','black','green','blue','pink','orange']




_X, _y = load_dataset(1, actions) # N, 30, 258
X, _y = unpack_dataset(_X,_y) #, N*30, 258
c = [colors[actions.index(yy)] for yy in _y ]
y = np.array([actions.index(yy) for yy in _y ])


# colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
#           "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]


XX = []
for _x in X:
    p, l, r = split_keypoints(_x)
    nright = translate(r)
    npose = translate(p)
    snp, snr = scaling(npose, nright)
    XX.append(snr)




figure, axesSubplot = plt.subplots() 
digits_tsne = TSNE(n_components = 2).fit_transform(XX, y)

for i in range(len(actions)):
    xy1= digits_tsne[np.where(y==i)]
    color = colors[i]
    axesSubplot.scatter(xy1[:, 0], xy1[:, 1], c=color) 
axesSubplot.legend(actions)

# print(digits_tsne.shape)

# axesSubplot.scatter(digits_tsne[:, 0], digits_tsne[:, 1], c=c) 
# axesSubplot.legend(['First line', 'Second line'])
# print('scatters')
axesSubplot.set_xticks(()) 
axesSubplot.set_yticks(()) 
plt.show()

# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()