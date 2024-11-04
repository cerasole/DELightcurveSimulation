import numpy as np

def compute_delay_matrix (t_A, t_B):
    """
    Function to evaluate the matrix of the delays.
    Arguments:
     - t_A, t_B: array-like
       Arrays of the times of the two lightcurves
    """
    # Definition of the matrix of the delays
    N_A, N_B = len(t_A), len(t_B)
    tau_ij = np.matmul(t_A.reshape(N_A, 1), np.ones(N_B).reshape(1, N_B)) \
           - np.matmul(np.ones(N_A).reshape(N_A, 1), t_B.reshape(1, N_B))
    return tau_ij


def compute_zDCF (x, y, r_sigma = 0.05):
    """
    Function to compute the zDCF of two vectors x and y.
    In the zDCF algorithm, the x and y vectors are belonging to the same time-lag bin.
    The function implements formula (3), (4) and (5) of the Tal Alexander 2013 paper.
    Moreover, it handles the limitation of the z value being defined for -1. < r < 1., while the uncertainties and
    formulas may return r slightly outside these limits. In this case, you can decide to return a corrected-z value,
    computed for r - r_sigma (if r >= 1.) or for r + r_sigma (if r <= -1.)
    Arguments:
     - x and y, array-like
      The two arrays to which the correlation is computed.
     - r_sigma, float
      If the correlation parameter r < 1 and r > -1, the z value is always defined.
      If this is not the case, I can check if r-r_sigma < 1 (or r+r_sigma > -1) and return that value, instead of something fixed.
      If r-r_sigma is still outside the allowed range, a custom z is returned (the one corresponding to r = +/- 0.99).
    Output:
     - z and r: values of the zDCF (Fisher's z-tranform of the correlation parameter r) and r
    """

    n = len(x)
    r = (1 / (n-1)) * \
        np.sum( (x - np.average(x)) * (y - np.average(y)) ) * \
        1 / (np.std(x) * np.std(y))

    if r < 1. and r > -1.:
        z = np.arctanh(r)
    # Handling of the case in which r is not strictly within the (-1., 1.) range (bin extremes are excluded)
    elif r >= 1.:
        print (f"The correlation coefficient value was equal to {r}, for which the corresponding z (= np.arctanh(r)) diverges.")
        if r - r_sigma < 1:
            z = np.arctanh(r - r_sigma)
            print (f"The z value corresponding to r - {r_sigma} = {r-r_sigma}, i.e. {z}, was returned instead.")
        else:
            z = np.arctanh(0.99)
            print (f"The z-value corresponding to r = 0.99, i.e. {z}, was returned instead.")
        print ("Please use the Single.do_zdcf_plot(r = True) for checking the correlation parameter r curve.")
    elif r <= -1.:
        print (f"The correlation coefficient value was equal to {r}, for which the corresponding z (= np.arctanh(r)) diverges.")
        if r + r_sigma > -1.:
            z = np.arctanh(r + r_sigma)
            print (f"The z value corresponding to r + {r_sigma} = {r + r_sigma}, i.e. {z}, was returned instead.")
        else:
            z = np.arctanh(-0.99)
            print (f"The z-value corresponding to r = -0.99, i.e. {z}, was returned instead.")
        print ("Please use the Single.do_zdcf_plot(r = True) for checking the correlation parameter r curve.")

    return z, r
