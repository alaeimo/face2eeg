# face2eeg/core/data/cv.py

"""
GroupKFold split generation and visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

def get_cv_splits(n_points=40, n_splits=5):
    X = np.ones(n_points)
    y = None
    groups = np.arange(n_points)
    cv = GroupKFold(n_splits=n_splits)
    training_sets = []
    testing_sets = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
      training_sets.append(train_idx)
      testing_sets.append(test_idx)
    return training_sets, testing_sets, cv

def plot_cv_indices(cv, X, y, group, ax, lw=10):
    n_split = cv.get_n_splits(X, y, group)
    cmap_data = plt.cm.viridis
    cmap_cv   = plt.cm.coolwarm

    # Generate the training/testing visualizations for each CV split
    for i, (train, test) in enumerate(cv.split(X=X, y=y, groups=group)):
        indices = np.array([np.nan] * len(X))
        indices[train] = 1
        indices[test]  = 0
        ax.scatter(range(len(indices)), [i + 0.5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data groups at the end
    ax.scatter(range(len(X)), [i + 1.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_split)) + ['group']
    ax.set_yticks(np.arange(n_split + 1) + 0.5)
    ax.set_yticklabels(yticklabels)
    ax.set(xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_split + 1.5, -.2], xlim=[-1, len(X)])
    ax.set_title(f'{type(cv).__name__}', fontsize=15)
    return ax