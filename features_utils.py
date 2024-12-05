from scipy.stats import median_abs_deviation
import numpy as np

def kde(features):
    """Computes the probability density function of the input signal using a Gaussian KDE (Kernel Density Estimate)
    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values
    """
    features_ = np.copy(features)
    xx = create_xx(features_)

    if min(features_) == max(features_):
        noise = np.random.randn(len(features_)) * 0.0001
        features_ = np.copy(features_ + noise)

    kernel = scipy.stats.gaussian_kde(features_, bw_method='silverman')

    return np.array(kernel(xx) / np.sum(kernel(xx)))

def gaussian(features):
    """Computes the probability density function of the input signal using a Gaussian function
    Parameters
    ----------
    features : nd-array
        Input from which probability density function is computed
    Returns
    -------
    nd-array
        probability density values
    """

    features_ = np.copy(features)

    xx = create_xx(features_)
    std_value = np.std(features_)
    mean_value = np.mean(features_)

    if std_value == 0:
        return 0.0
    pdf_gauss = scipy.stats.norm.pdf(xx, mean_value, std_value)

    return np.array(pdf_gauss / np.sum(pdf_gauss))

def entropy(signal, prob='standard'):
    """Computes the entropy of the signal using the Shannon Entropy.
    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)
    Returns
    -------
    float
        The normalized entropy value
    """

    if prob == 'standard':
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == 'kde':
        p = kde(signal)
    elif prob == 'gauss':
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return - np.sum(p * np.log2(p)) / np.log2(len(signal))
    
def mean_abs_deviation(signal):
    """Computes mean absolute deviation of the signal.
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which mean absolute deviation is computed
    Returns
    -------
    float
        Mean absolute deviation result
    """
    return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)

def median_abs_deviation(signal, scale):
    """Computes median absolute deviation of the signal.
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed
    Returns
    -------
    float
        Mean absolute deviation result
    """
    return median_abs_deviation(signal, scale)