import numpy as np


def generate_GBM(mu, sigma, n=50, dt=0.1, x_0=10, random_seed=None):

    if not random_seed:
        normal_component = np.random.normal(0, np.sqrt(dt), size=n)
    else:
        np.random.seed(random_seed)
        normal_component = np.random.normal(0, np.sqrt(dt), size=n)

    x = np.exp((mu - sigma ** 2 / 2) * dt + sigma * normal_component)

    x = np.concatenate([np.ones(1), x])
    x = x_0 * x.cumprod()

    return x


def NPV_check(row, threshold):

    if row['has_shirked']:
        return True
    else:
        return (row.iloc[:-1] <= threshold).sum() > 0
