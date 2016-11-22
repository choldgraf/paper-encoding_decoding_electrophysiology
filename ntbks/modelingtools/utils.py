"""Helper functions for encoding / decoding tutorials."""
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn import cross_validation as cv
from itertools import product
from skimage import filters as flt
from matplotlib.colors import LinearSegmentedColormap


# To reproduce results
np.random.seed(1337)

# Useful variables
colors_score = ['#67a9cf', '#ef8a62']
colors_activity = ['#7fbf7b', '#af8dc3']
cmap_activity = LinearSegmentedColormap.from_list(
    'activity', colors_activity)
cmap_score = LinearSegmentedColormap.from_list(
    'activity', colors_score)


def delay_time_series(X, delays, sfreq):
    """Delay a time series.

    Parameters
    ----------
    X : array, shape (n_epochs, n_features, n_times)
        The data to delay.
    delays : array, shape (n_delays,)
        The delays (in seconds) to create.
    sfreq : float
        The sampling frequency of the data.

    Returns
    -------
    X_delayed : array, shape (n_epochs, n_features, n_delays, n_times)
        The delayed data
    """
    delays_ixs = (delays * sfreq).astype(int)
    X_delayed = np.zeros([X.shape[0], X.shape[1], len(delays), X.shape[-1]])

    for ii, iep in enumerate(X):
        for jj, idelay in enumerate(delays_ixs):
            i_delayed = np.roll(iep, -idelay, axis=-1)
            if idelay <= 0:
                i_slice = slice(-idelay, None)
            else:
                i_slice = slice(None, -idelay)
            X_delayed[ii, :, jj, i_slice] = i_delayed[..., i_slice]
    return X_delayed


def plot_activity_on_brain(activity, im, x, y, size_scale=1e4, ax=None,
                           cmap=None, with_cbar=True, **kwargs):
    """Plots activity for electrodes on a brain."""
    if ax is None:
        fig, ax = plt.subplots()
    cmap = plt.cm.coolwarm if cmap is None else cmap
    ax.imshow(im)
    ax.scatter(x, y, s=np.abs(activity) * size_scale, c=activity,
               cmap=cmap, **kwargs)
    ax.set_title('Channel Loadings')
    ax.set_axis_off()

    if with_cbar is True:
        y_colorscatter = np.linspace(0, im.shape[0] * .8, 20)
        act_colorscatter = (y_colorscatter - np.mean(y_colorscatter))[::-1]
        ax.scatter([im.shape[1] + 50]*len(y_colorscatter), y_colorscatter,
                   c=act_colorscatter, s=np.abs(act_colorscatter) * 5,
                   cmap=cmap)
    return ax


def plot_cv_indices(cv, ax=None):
    """Create a sample plot for indices of a cross-validation object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    max_index = np.max(np.hstack([tr for tr, tt in cv])) + 1
    n_cv_iterations = len(cv)
    zeros = np.zeros([max_index, n_cv_iterations])
    for ii, (tr, tt) in enumerate(cv):
        zeros[tt, ii] = 1

    ax.pcolormesh(range(n_cv_iterations), range(max_index),
                  zeros, cmap='Greys')
    ax.set_ylabel('Sample index')
    ax.set_title('{} cross validation over trials '
                 '(black = test set)'.format(type(cv).__name__))
    plt.tight_layout()
    return ax


def cross_validate_alpha(X, y, cv_outer, alphas, n_cv_inner=5):
    scores = np.zeros([len(cv_outer), n_cv_inner, len(alphas)])
    coefs = np.zeros([len(cv_outer), n_cv_inner,
                      len(alphas), X.shape[1]])
    pbar = mne.utils.ProgressBar(len(cv_outer))
    for ii, (tr, tt) in enumerate(cv_outer):
        # Split trials into train/test sets
        outer_y_tr = y[tr]
        outer_y_tt = y[tt]
        outer_X_tr = X[tr]
        outer_X_tt = X[tt]

        # Now within the outer loop, do an inner loop for the ridge parameter
        cv_inner = cv.KFold(outer_y_tr.shape[0], n_folds=n_cv_inner,
                            shuffle=True)
        for jj, (tr_val, tt_val) in enumerate(cv_inner):
            # Create our training / testing data for the inner loop
            inner_y_tr = np.hstack(outer_y_tr[tr_val]).T
            inner_y_tt = np.hstack(outer_y_tr[tt_val]).T

            inner_X_tr = np.hstack(outer_X_tr[tr_val]).T
            inner_X_tt = np.hstack(outer_X_tr[tt_val]).T

            inner_X_tr = scale(inner_X_tr)
            inner_X_tt = scale(inner_X_tt)
            inner_y_tr = scale(inner_y_tr)
            inner_y_tt = scale(inner_y_tt)

            # For each alpha value, fit / score the model
            for kk, alpha in enumerate(alphas):
                model = Ridge(alpha=alpha)
                model.fit(inner_X_tr, inner_y_tr)
                score = r2_score(inner_y_tt, model.predict(inner_X_tt))
                scores[ii, jj, kk] = score
                coefs[ii, jj, kk] = model.coef_
        pbar.update(ii + 1)
    return scores, coefs


def plot_gabors_2d(n_freqs=4, n_angles=7, fmin=.02, fmax=.2, sigma=15,
                   ax_size=.5, background_extend=1, shrink_radius_mask=10,
                   figsize=(6, 2)):
    # Create our angles / radii
    dists = np.linspace(0, 1, n_freqs)
    angle = np.linspace(0, np.pi, n_angles)
    pairs = np.array(list(product(angle, dists)))

    # Convert to X/Y and center the X-axis
    x, y = pol2cart(*pairs.T)
    x = x / 2. + .5

    # Create figure and axes
    fig = plt.figure(figsize=figsize)
    irho_plotted = False
    for iphi, irho in pairs:
        if irho_plotted is True and irho == 0:
            continue
        irho_plotted = True
        ix, iy = pol2cart(iphi, irho)
        ix = (ix + 1) / 2.   # Convert to 0 to 1
        # Create the filter
        frequency = irho * (fmax - fmin) + fmin
        kernel = np.real(flt.gabor_kernel(frequency, theta=iphi,
                                          sigma_x=sigma, sigma_y=sigma))

        # Create a mask for the filter
        # Center of mask
        a, b = np.array(kernel.shape) // 2
        # Total size of space
        n = kernel.shape[0]
        # Radius of mask
        r = a - shrink_radius_mask
        y, x = np.ogrid[-a:n-a, -b:n-b]
        mask = x*x + y*y <= r*r
        kernel[~mask] = np.nan
        background = np.zeros_like(kernel)
        mask_back = x*x + y*y <= (r+background_extend) * (r+background_extend)
        background[mask_back] = 1.
        background[~mask_back] = np.nan

        # Plot the filter at specified location
        vmin = kernel[~np.isnan(kernel)].max()
        ax = plt.axes((1-ix, iy, ax_size, ax_size))
        ax.imshow(background, cmap='Greys_r', interpolation='hanning')
        ax.imshow(kernel, cmap='Greys',
                  vmin=-vmin, vmax=vmin, interpolation='hanning')

    for ax in fig.axes:
        ax.set_axis_off()
    return fig


def pol2cart(phi, rho):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
