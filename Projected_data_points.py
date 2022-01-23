import numpy as np
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from myPCA import myPCA, ProjectDatapoints

X,r = datasets.load_digits(return_X_y=True)

N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X

W,mu = myPCA(X,2)

X_new = ProjectDatapoints(X,W,mu)


# plot the projected digits on R^2
fig,ax = plt.subplots(1,1,figsize=(6, 6))

cmap= plt.cm.get_cmap('tab10')

for i in range(10):
    x_i = X_new[r==i]
    ax.scatter(x_i[:,0],x_i[:,1],color=cmap(i),label='{}'.format(i))

ax.legend()
fig.savefig('myPCA.png')
