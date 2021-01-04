#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yellowbrick


# In[2]:


# import packages needed for the procedure
import pandas as pd

# read data as data
data = pd.read_csv("Mroz.csv")

# check the dimension of the table
print("The dimension of the table is: ", data.shape)


# In[3]:


data.describe()


# In[4]:


data.describe(include=['O'])


# In[5]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lfp', 'wc', 'hc']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[6]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lfp', 'wc', 'hc' 'lwg']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[ ]:





# In[7]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 2, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)


# In[8]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[9]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lwg', 'wc', 'hc', 'lfp', 'inc']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[10]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lwg', 'lfp', 'inc', 'wc', 'hc']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[11]:


# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# Specify the features of interest
num_features = ['age', 'lwg', 'inc', 'lfp', 'wc', 'hc']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)


# In[12]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)


# In[13]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_k5, Y_k5)
axes[1, 1].set_title('k5', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

 make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_k618, Y_k618)
axes[1, 1].set_title('k618', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)


# In[14]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_k5, Y_k5)
axes[1, 1].set_title('k5', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_k618, Y_k618)
axes[1, 1].set_title('k618', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)


# In[15]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 0].bar(X_k5, Y_k5)
axes[2, 0].set_title('k5', fontsize=25)
axes[2, 0].set_ylabel('Counts', fontsize=20)
axes[2, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 1].bar(X_k618, Y_k618)
axes[2, 1].set_title('k618', fontsize=25)
axes[2, 1].set_ylabel('Counts', fontsize=20)


# In[16]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 0].bar(X_k5, Y_k5)
axes[2, 0].set_title('k5', fontsize=25)
axes[2, 0].set_ylabel('Counts', fontsize=20)
axes[2, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 1].bar(X_k618, Y_k618)
axes[2, 1].set_title('k618', fontsize=25)
axes[2, 1].set_ylabel('Counts', fontsize=20)
axes[2, 1].tick_params(axis='both', labelsize=15)


# In[17]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].as_matrix()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[18]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 0].bar(X_k5, Y_k5)
axes[2, 0].set_title('k5', fontsize=25)
axes[2, 0].set_ylabel('Counts', fontsize=20)
axes[2, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 1].bar(X_k618, Y_k618)
axes[2, 1].set_title('k618', fontsize=25)
axes[2, 1].set_ylabel('Counts', fontsize=20)
axes[2, 1].tick_params(axis='both', labelsize=15)



# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].as_matrix()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[19]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 2)

# make the data read to feed into the visulizer
X_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['lfp']
Y_lfp = data.replace({'lfp': {1: 'yes', 0: 'no'}}).groupby('lfp').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 0].bar(X_lfp, Y_lfp)
axes[0, 0].set_title('lfp', fontsize=25)
axes[0, 0].set_ylabel('Counts', fontsize=20)
axes[0, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_inc = data.groupby('inc').size().reset_index(name='Counts')['inc']
Y_inc = data.groupby('inc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[0, 1].bar(X_inc, Y_inc)
axes[0, 1].set_title('inc', fontsize=25)
axes[0, 1].set_ylabel('Counts', fontsize=20)
axes[0, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_wc = data.groupby('wc').size().reset_index(name='Counts')['wc']
Y_wc = data.groupby('wc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 0].bar(X_wc, Y_wc)
axes[1, 0].set_title('wc', fontsize=25)
axes[1, 0].set_ylabel('Counts', fontsize=20)
axes[1, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_hc = data.groupby('hc').size().reset_index(name='Counts')['hc']
Y_hc = data.groupby('hc').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[1, 1].bar(X_hc, Y_hc)
axes[1, 1].set_title('hc', fontsize=25)
axes[1, 1].set_ylabel('Counts', fontsize=20)
axes[1, 1].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k5 = data.groupby('k5').size().reset_index(name='Counts')['k5']
Y_k5 = data.groupby('k5').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 0].bar(X_k5, Y_k5)
axes[2, 0].set_title('k5', fontsize=25)
axes[2, 0].set_ylabel('Counts', fontsize=20)
axes[2, 0].tick_params(axis='both', labelsize=15)

# make the data read to feed into the visulizer
X_k618 = data.groupby('k618').size().reset_index(name='Counts')['k618']
Y_k618 = data.groupby('k618').size().reset_index(name='Counts')['Counts']
# make the bar plot
axes[2, 1].bar(X_k618, Y_k618)
axes[2, 1].set_title('k618', fontsize=25)
axes[2, 1].set_ylabel('Counts', fontsize=20)
axes[2, 1].tick_params(axis='both', labelsize=15)


# In[20]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].as_matrix()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[21]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[22]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].df.to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[23]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[24]:



# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = df.to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[25]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = DataFrame.to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[26]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[27]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# Load the credit dataset
X, y = pd.read_csv("Mroz.csv")

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()    


# In[28]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# Load the credit dataset
X = data[num_features].as_matrix()

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()    


# In[29]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D
Import numpy as np

# extract the numpy arrays from the data frame
X = data[num_features].to_numpy()

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data


# In[30]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = data[num_features].as_matrix()

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[31]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = pd.read_csv('Mroz.csv')

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[32]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = pd.read_csv('Mroz.csv')

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[33]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X, y = pd.read_csv('Mroz.csv')

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[34]:


type(data.to_nump())


# In[35]:


type(data.to_numpy())


# In[36]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = numpy.ndarray

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[37]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = type(data.to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[38]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = type(data.to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[39]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X, y = type(data.to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X, y)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.show()              # Finalize and render the figure


# In[40]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = type(data.to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[41]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
X = type(data[num_features].to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[42]:


from yellowbrick.features import Rank1D
# Instantiate the 1D visualizer with the Sharpiro ranking algorithm
visualizer = Rank1D(features=feature_names, algorithm='shapiro')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # visualise


# In[43]:


from yellowbrick.features import Rank1D
# Instantiate the 1D visualizer with the Sharpiro ranking algorithm
visualizer = Rank1D(features=num_features, algorithm='shapiro')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # visualise


# In[44]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = df[num_features]


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=feature_names, algorithm='covariance') 
#visualizer = Rank2D(features=feature_names, algorithm='pearson')
visualizer.fit(X)                
visualizer.transform(X)             
visualizer.poof()


# In[45]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = num_features


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=feature_names, algorithm='covariance') 
#visualizer = Rank2D(features=feature_names, algorithm='pearson')
visualizer.fit(X)                
visualizer.transform(X)             
visualizer.poof()


# In[46]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = df[num_features]


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='covariance') 
#visualizer = Rank2D(features=feature_names, algorithm='pearson')
visualizer.fit(X)                
visualizer.transform(X)             
visualizer.poof()


# In[47]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = num_features


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='covariance') 
#visualizer = Rank2D(features=feature_names, algorithm='pearson')
visualizer.fit(X)                
visualizer.transform(X)             
visualizer.poof()


# In[48]:



num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = df[num_features]


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[49]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = num_features


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[50]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = data[num_features]


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[51]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = data[num_features].as_numpy()


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[52]:


num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = data[num_features].to_numpy()


from yellowbrick.features import Rank2D
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[53]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']
X = type(data[num_features].to_numpy())


from yellowbrick.features import Rank2D

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[54]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

X = type(data[num_features].to_numpy())
# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[55]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

X = data[num_features].to_numpy()

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[56]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

X = data[num_features].to_numpy()

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='covariance')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[57]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)

from yellowbrick.features import Rank2D

X = type(data[num_features].to_numpy())

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='covariance')

visualizer.fit(X)           # Fit the data to the visualizer
visualizer.transform(X)        # Transform the data
visualizer.poof()              # Finalize and render the figure


# In[58]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 50

# setup the color for yellowbrick visulizer
from yellowbrick.style import set_palette
set_palette('sns_bright')

# import packages
from yellowbrick.features import ParallelCoordinates
# Specify the features of interest and the classes of the target
classes = ['No', 'Yes']
num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']

# copy data to a new dataframe
data_norm = data.copy()
# normalize data to 0-1 range
for feature in num_features:
    data_norm[feature] = (data[feature] - data[feature].mean(skipna=True)) / (data[feature].max(skipna=True) - data[feature].min(skipna=True))

# Extract the numpy arrays from the data frame
X = data_norm[num_features].as_matrix()
y = data.Survived.as_matrix()

# Instantiate the visualizer
# Instantiate the visualizer
visualizer = ParallelCoordinates(classes=classes, features=num_features)


visualizer.fit(X, y)      # Fit the data to the visualizer
visualizer.transform(X)   # Transform the data
visualizer.poof()         # Draw/show/poof the data


# In[59]:


# set up the figure size
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (15, 7)
plt.rcParams['font.size'] = 50

# setup the color for yellowbrick visulizer
from yellowbrick.style import set_palette
set_palette('sns_bright')

# import packages
from yellowbrick.features import ParallelCoordinates
# Specify the features of interest and the classes of the target
classes = ['1', '2']
num_features = ['age', 'lfp', 'wc', 'hc', 'lwg']

# copy data to a new dataframe
data_norm = data.copy()
# normalize data to 0-1 range
for feature in num_features:
    data_norm[feature] = (data[feature] - data[feature].mean(skipna=True)) / (data[feature].max(skipna=True) - data[feature].min(skipna=True))

# Extract the numpy arrays from the data frame
X = data_norm[num_features].as_matrix()
y = data.Survived.as_matrix()

# Instantiate the visualizer
# Instantiate the visualizer
visualizer = ParallelCoordinates(classes=classes, features=num_features)


visualizer.fit(X, y)      # Fit the data to the visualizer
visualizer.transform(X)   # Transform the data
visualizer.poof()         # Draw/show/poof the data


# In[60]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# In[61]:


dataset= pd.read_csv("Mroz.csv")


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set() 

#-------------------Survived/Died by Class -------------------------------------
working_class = dataset[dataset['lfp']==yes]['inc'].value_counts()
nonworking_class = dataset[dataset['lfp']==no]['inc'].value_counts()
df_class = pd.DataFrame([working_class,nonworking_class])
df_class.index = ['Working','Unemployed']
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Income")

# display table
from IPython.display import display
display(df_class)


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set() 

#-------------------Survived/Died by Class -------------------------------------
working_class = dataset[dataset['lfp']=='yes']['inc'].value_counts()
nonworking_class = dataset[dataset['lfp']=='no']['inc'].value_counts()
df_class = pd.DataFrame([working_class,nonworking_class])
df_class.index = ['Working','Unemployed']
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Income")

# display table
from IPython.display import display
display(df_class)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set() 

#-------------------Survived/Died by Class -------------------------------------
working_class = dataset[dataset['lfp']=="yes"]['inc'].value_counts()
nonworking_class = dataset[dataset['lfp']=="no"]['inc'].value_counts()
df_class = pd.DataFrame([working_class,nonworking_class])
df_class.index = ['Working','Unemployed']
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Income")

# display table
from IPython.display import display
display(df_class)


# In[65]:


#-------------------Working/Unemployed by WC------------------------------------
   
Working = dataset[dataset.Working == 1]['wc'].value_counts()
Unemployed = dataset[dataset.Working == 0]['wc'].value_counts()
df_wc = pd.DataFrame([Working , Unemployed])
df_wc.index = ['Working','Unemployed']
df_wc.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Wife's College Attendance")


wc_working= df_wc.working[0]/df_wc.working.sum()*100
wc_unemployed = wc_wc.unemployed[0]/df_wc.unemployed.sum()*100
print("Percentage of Employed with Wife's College Attendance:" ,round(wc_working), "%")
print("Percentage of Unemployed with Wife's College Attendance:" ,round(wc_unemployed), "%")

# display table
from IPython.display import display
display(df_wc) 


# In[66]:


#-------------------Working/Unemployed by WC------------------------------------
   
Working = dataset[dataset.Working == 1]['wc'].value_counts()
Unemployed = dataset[dataset.Working == 0]['wc'].value_counts()
df_wc = pd.DataFrame([Yes , No])
df_wc.index = ['Yes','No']
df_wc.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Wife's College Attendance")


wc_working= df_wc.working[0]/df_wc.working.sum()*100
wc_unemployed = wc_wc.unemployed[0]/df_wc.unemployed.sum()*100
print("Percentage of Employed with Wife's College Attendance:" ,round(wc_working), "%")
print("Percentage of Unemployed with Wife's College Attendance:" ,round(wc_unemployed), "%")

# display table
from IPython.display import display
display(df_wc) 


# In[67]:


#-------------------Working/Unemployed by WC------------------------------------
   
Working = dataset[dataset.Yes == 1]['wc'].value_counts()
Unemployed = dataset[dataset.No == 0]['wc'].value_counts()
df_wc = pd.DataFrame([Yes , No])
df_wc.index = ['Yes','No']
df_wc.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Wife's College Attendance")


wc_working= df_wc.working[0]/df_wc.working.sum()*100
wc_unemployed = wc_wc.unemployed[0]/df_wc.unemployed.sum()*100
print("Percentage of Employed with Wife's College Attendance:" ,round(wc_working), "%")
print("Percentage of Unemployed with Wife's College Attendance:" ,round(wc_unemployed), "%")

# display table
from IPython.display import display
display(df_wc) 


# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn
seaborn.set() 

#-------------------Survived/Died by Class -------------------------------------
working_class = dataset[dataset['lfp']=='yes']['inc'].value_counts()
nonworking_class = dataset[dataset['lfp']=='no']['inc'].value_counts()
df_class = pd.DataFrame([working_class,nonworking_class])
df_class.index = ['Working','Unemployed']
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Working/Unemployed by Income")

# display table
from IPython.display import display
display(df_class)


# In[69]:


data.head()


# In[70]:


data.describe()


# In[71]:


data['Unemployed'] = 'no' - data['lfp']
data.groupby('inc').agg('sum')[['lfp', 'Unemployed']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r']);


# In[72]:


data['lfp'] = "no" - data['lfp']
data.groupby('inc').agg('sum')[['lfp', 'lfp']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r']);


# In[73]:


sample.lfp.map(dict(yes=1, no=0))


# In[74]:


lfp.map(dict(yes=1, no=0))


# In[75]:



data['lfp'] == 'no' - data['lfp']
data.groupby('inc').agg('sum')[['lfp', 'lfp']].plot(kind='bar', figsize=(25, 7),
                                                          stacked=True, colors=['g', 'r']);


# In[76]:


fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='age', y='inc', 
               hue='lfp', data=data, 
               split=True,
               palette={no: "r", yes: "g"}
              );


# In[77]:


import matplotlib.pyplot as plt
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[78]:


data.corr()


# In[79]:


from pandas.plotting import scatter_matrix
scatter_matrix(data)


# In[80]:


attributes=["k5", "k618", "inc"] 
scatter_matrix(data[attributes], figsize=(12,8)


# In[81]:


attributes=["k5", "k618", "inc"] 
scatter_matrix(data[attributes])


# In[82]:


data.plot(kind="scatter", x="inc", y="lfp", alpha=0.2)


# In[83]:


data.plot(kind="scatter", x="inc", y="lfp")


# In[84]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data[["lfp","k5","k618","wc","hc","lwg","inc"]].corr(), annot = True)
plt.show()


# In[85]:


sns.factorplot(x="lfp", y ="inc", data=data, kind="bar", size=3)
plt.show()


# In[86]:


data.hist(bins=20, figsize=(10,10))
plt.show()


# In[87]:


sns.distplot(data[“inc”])
plt.show()


# In[88]:


sns.distplot(data['inc'])
plt.show()


# In[89]:


fig=plt.figure()


# In[90]:


sns.stripplot(x="lfp", y="inc", data=data)


# In[91]:


sns.swarmplot(x="lfp", y="inc", data=data)


# In[92]:


sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
sns.distplot(
    data['inc'], norm_hist=False, kde=False, bins=20, hist_kws={"alpha": 1}
).set(xlabel='inc', ylabel='Count');


# In[93]:


sns.countplot(data['k5', 'k618']);


# In[94]:


sns.countplot(data['k5']);


# In[95]:


sns.countplot(data['k618']);


# In[96]:


sns.scatterplot(x=data['k618'], y=data['inc']);


# In[97]:


sns.scatterplot(x=data['k5', 'k618'], y=data['inc']);


# In[98]:


sns.scatterplot(x=data['lwg'], y=data['inc']);


# In[99]:


sns.scatterplot(x=data['inc'], y=data['lwg']);


# In[100]:


sns.scatterplot(x=data['lfp'], y=data['inc']);


# In[101]:


numerical = [
  'k5', 'k618', 'age', 'lwg', 'inc'
]
categorical = [
  'lfp', 'wc', 'hc'
]

data = data[numerical + categorical]
data.shape


# In[102]:


fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='inc', data=data, ax=subplot)


# In[103]:


fig, ax = plt.subplots(3, 0, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=housing, ax=subplot)


# In[104]:


fig, ax = plt.subplots(3, 1, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='SalePrice', data=housing, ax=subplot)


# In[105]:


fig, ax = plt.subplots(3, 1, figsize=(15, 10))
for var, subplot in zip(categorical, ax.flatten()):
    sns.boxplot(x=var, y='inc', data=data, ax=subplot)


# In[106]:


sorted_nb = housing.groupby(['k5'])['inc'].median().sort_values()
sns.boxplot(x=housing['k5'], y=housing['inc'], order=list(sorted_nb.index))


# In[107]:


sorted_nb = data.groupby(['k5'])['inc'].median().sort_values()
sns.boxplot(x=data['k5'], y=data['inc'], order=list(sorted_nb.index))


# In[108]:


sorted_nb = data.groupby(['k5'], ['k618'])['inc'].median().sort_values()
sns.boxplot(x=data['k5'], y=data['inc'], order=list(sorted_nb.index))


# In[109]:


sorted_nb = data.groupby(['k5'])['inc'].median().sort_values()
sns.boxplot(x=data['k5'], y=data['inc'], order=list(sorted_nb.index))


# In[110]:


sorted_nb = data.groupby(['k5'])['inc'].median().sort_values()
sns.boxplot(x=data['k5', 'k618'], y=data['inc'], order=list(sorted_nb.index))


# In[111]:


df.head()


# In[112]:


data.head()


# In[113]:


sns.lmplot(x='lwg', y='inc', data=data)


# In[114]:


sns.lmplot(x='lwg', y='inc', data=data,
           fit_reg=False, # No regression line
           hue='Stage')   # Color by evolution stage


# In[115]:


sns.lmplot(x='lwg', y='inc', data=data,
           fit_reg=False, # No regression line


# In[116]:


sns.lmplot(x='lwg', y='inc', data=data,
           fit_reg=False


# In[117]:


sns.boxplot(data=data)


# In[118]:



stats_data = data.drop(['k5', 'k618', 'lwg'], axis=1)
 

sns.boxplot(data=stats_data)


# In[119]:


data.head()


# In[120]:


data.describe()


# In[121]:


sns.scatterplot(x=data['age'], y=data['inc']);


# In[122]:


sns.scatterplot(x=data['inc'], y=data['age']);


# In[123]:


sns.pairplot(data)
sns.plt.show()


# In[124]:


import matplotlib.pyplot as plt
sns.pairplot(data)
sns.plt.show()


# In[125]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data)
sns.plt.show()


# In[1]:


sns.relplot(x="k5", y="inc", data=tips);


# In[2]:


data.relplot(x="k5", y="inc", data=tips);


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


# In[4]:


sns.relplot(x="k5", y="inc", data=tips);


# In[5]:


sns.relplot(x="k5", y="inc", data=data);


# In[6]:


sns.countplot(data['k5']);


# In[7]:


data = pd.read_csv("Mroz.csv")


# In[8]:


sns.countplot(data['k5']);


# In[9]:


sns.relplot(x="k5", y="inc", data=data);


# In[10]:


sns.relplot(x="inc", y="lwg", data=data);


# In[11]:


sns.relplot(x="lwg", y="inc", hue"lfp", data=data);


# In[12]:


sns.relplot(x="lwg", y="inc", hue="lfp", data=data);


# In[13]:


data = data.rename(columns={'lfp': 'Labor Force Participation', 'wc' : 'Wife in College', 'hc' : 'Husband in College', 'lwg' : 'Log Expected Wage Rate', 'inc' : 'Income'})


# In[14]:


sns.relplot(x="lwg", y="inc", hue="lfp", data=data);


# In[15]:


sns.relplot(x="Log Expected Wage Rate", y="Income", hue="Labor Force Participation", data=data);


# In[16]:


sns.relplot(x="Income", y="Log Expected Wage Rate", hue="Labor Force Participation", data=data);


# In[17]:


sns.relplot(x="age", y="k5", hue="Labor Force Participation", data=data);


# In[18]:


sns.relplot(x="age", y="Income", hue="Labor Force Participation", data=data);


# In[19]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[20]:


data = data.rename(columns={Labor Force Participation' : 'lfp', 'Wife in College' : 'wc', 'Husband in College' : 'hc', 'Log Expected Wage Rate' : 'lwg', 'Income' : 'inc'})


# In[21]:


data = data.rename(columns={'Labor Force Participation' : 'lfp', 'Wife in College' : 'wc', 'Husband in College' : 'hc', 'Log Expected Wage Rate' : 'lwg', 'Income' : 'inc'})


# In[22]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[23]:


data.head()


# In[24]:


pd.read_csv(io.StringIO(df.to_csv(index=False)))


# In[25]:


pd.read_csv('Mroz.csv', index_col=[0])


# In[26]:


data.head()


# In[27]:


data = pd.read_csv('Mroz.csv', index_col=[0])


# In[28]:


data.head()


# In[29]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[30]:


data.hist(bins=20, figsize=(10,10))
plt.show()


# In[31]:


sns.relplot(x="lwg", y="inc", hue="hc", data=data);


# In[32]:


sns.relplot(x="lwg", y="inc", hue="wc", data=data);


# In[ ]:


sns.factorplot(x='inc', col='lwg', kind='count', data=data)


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# read data as data
data = pd.read_csv('Mroz.csv', index_col=[0])

data. describe() 


# In[2]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[3]:


from numpy.random import RandomState
import pandas as pd
import numpy as np

data = pd.read_csv('Mroz.csv', index_col=[0])
rng = RandomState()

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train_ml = train.copy()
test_ml = test.copy()


# In[7]:


#pandas get_dummies for wc and hc

#for train_ml
train_ml = pd.get_dummies(train_ml, columns=['wc', 'hc'], drop_first=True)

#for test_ml
test_ml = pd.get_dummies(test_ml, columns=['wc', 'hc'], drop_first=True)


# In[8]:


train_ml.head()


# In[9]:


test_ml.head()


# In[10]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[11]:


#for train_ml
train_ml = pd.get_dummies(train_ml, columns=['wc', 'hc', 'lfp'], drop_first=True)

#for test_ml
test_ml = pd.get_dummies(test_ml, columns=['wc', 'hc', 'lfp'], drop_first=True)


# In[12]:


train_ml = train.copy()
test_ml = test.copy()


# In[13]:


#for train_ml
train_ml = pd.get_dummies(train_ml, columns=['wc', 'hc', 'lfp'], drop_first=True)

#for test_ml
test_ml = pd.get_dummies(test_ml, columns=['wc', 'hc', 'lfp'], drop_first=True)


# In[14]:


train_ml.head() 


# In[15]:


test_ml.head()


# In[16]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('lfp',axis=1), train_ml['lfp'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['lfp'], test_size=0.30, random_state=101)


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('inc',axis=1), train_ml['inc'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['inc'], test_size=0.30, random_state=101)


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml.drop('lfp',axis=1))
scaled_features = scaler.transform(train_ml.drop('lfp',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[20]:


train_ml.head()


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml.drop('lfp_yes',axis=1))
scaled_features = scaler.transform(train_ml.drop('lfp_yes',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[1]:


f,ax=plt.subplots(1,2,figsize=(16,7))
data['inc'][train['lfp']=='Yes'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
data['inc'][train['lfp']=='No'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Cheater (male)')
ax[1].set_title('Cheater (female)')

plt.show()


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


f,ax=plt.subplots(1,2,figsize=(16,7))
data['inc'][train['lfp']=='Yes'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
data['inc'][train['lfp']=='No'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Cheater (male)')
ax[1].set_title('Cheater (female)')

plt.show()


# In[ ]:




