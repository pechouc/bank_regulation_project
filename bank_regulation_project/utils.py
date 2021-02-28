"""
This module provides several functions useful to run the simulations that we define in the economy.py file.

It notably includes the generate_GBM function, which is used to simulate sample paths of a geometric Brownian motion.

The latter was inspired by the code snippet provided on the Wikipedia page about geometric Brownian motions.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT(S)

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# CONTENT

def generate_GBM(mu, sigma, n=50, dt=0.1, x_0=10, random_seed=None):
    """
    This function is used to simulate a geometric Brownian motion. It was inspired by the code snippet provided on the
    Wikipedia page about geometric Brownian motions (see section 'Simulating sample paths').

    It requires several arguments:

    - mu: the instantaneous drift of the geometric Brownian motion;

    - sigma: its instantaneous variance;

    - n: the number of timesteps to simulate beside the starting point x_0 (see below);

    - dt: as in the formulas for the geometric Brownian motion, this argument can be interpreted as the length of each
    timestep at which we simulate the motion. This, combined with the number of periods, is necessary for a programmatic
    approximation of continuous time implied by the Brownian motion;

    - x_0: the initial value from which the simulation starts;

    - random_seed: this argument may contain any natural integer which will determine the "state of the world" in which
    the simulation is assumed to occur. As a consequence, calling the generate_GBM function several times with the same
    random_seed will yield the same sample paths each time.

    Therefore, the function returns an array that contains (n+1) points, including the initial value x_0, and constitu-
    tes the simulated path for the geometric Brownian motion.
    """

    # We distinguish two cases based on whether a random_seed was passed as an argument or not
    if not random_seed:
        # If no random_seed was set, we run n simulations of a centered normal distribution of variance dt
        normal_component = np.random.normal(0, np.sqrt(dt), size=n)

    else:
        # If a random_seed was passed, we first use it to fix the random state
        np.random.seed(random_seed)
        # Then, we run n simulations of a centered normal distribution of variance dt
        normal_component = np.random.normal(0, np.sqrt(dt), size=n)

    # We multiply each simulation of the centered normal distribution by sigma, we add to it (mu - sigma ** 2 / 2) * dt
    # (this transformation comes from Itô's lemma) and we eventually apply the exponential to the resulting quantity
    x = np.exp((mu - sigma ** 2 / 2) * dt + sigma * normal_component)

    # We add a 1 at the beginning of the array that contains the x's (which will correspond to the initial value)
    x = np.concatenate([np.ones(1), x])

    # We multiply each element in the array by the initial value, x_0
    x = x_0 * x.cumprod()

    return x


def NPV_check(row, threshold, under_macro_shock=False, column_indices=None):
    """
    This function is a very brief one used in the apply method of the simulation DataFrame in the run_simulation method
    of an Economy instance (cf. the economy.py file).

    It takes two arguments:

    - row: the row of the simulation DataFrame considered for the check (the apply method, set with the axis=1 argument,
    automatically iterates over the rows of the DataFrame);

    - threshold: the positive expected net present value threshold under which a bank using the good asset monitoring
    technology is not expected to be able to compensate for the monitoring cost in the long run.

    The function returns a boolean:

    - True, if the bank has chosen the bad asset monitoring technology at some point in time or if it uses the good te-
    chnology but its cash flows have gone below the positive NPV threshold;

    - False, if none of the two conditions is met.
    """
    if not under_macro_shock:
        if row['has_shirked']:
            return True
        else:
            return (row.iloc[:-1] <= threshold).sum() > 0

    if under_macro_shock:
        if row['has_shirked_under_shock']:
            return True
        else:
            return (row.loc[column_indices] <= threshold).sum() > 0


def get_a_exponent(mu, sigma, r):
    """
    This function allows to compute the a_G or a_B exponent defined in the paper, at page 139 for the first time.

    It requires as arguments:

    - mu: the instantaneous drift of the geometric Brownian motion considered;

    - sigma: the instantaneous variance of the geometric Brownian motion considered;

    - r: the interest rate assumed for the economy.

    It then computes and returns the scalar used for instance in the computation of regulator's liquidation thresholds.
    """
    return (1 / 2) + (mu / (sigma ** 2)) + np.sqrt(((mu / (sigma ** 2)) - (1 / 2)) ** 2 + (2 * r) / (sigma ** 2))
