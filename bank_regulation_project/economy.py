import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import generate_GBM, NPV_check


class TypicalBank:

    def __init__(self, x_0, b, r, mu_G, sigma_G, mu_B, sigma_B):
        """
        This is the instantiation method for the TypicalBank class.

        It requires as arguments:

        - x_0: the initial value of the bank's cash flows;

        - b: the monitoring cost associated with the good asset management technology;

        - r: the interest rate;

        - mu_G: the instantaneous drift associated with the good asset management technology;

        - mu_B: the instantaneous drift associated with the bad asset management technology;

        - sigma_G: the instantaneous variance associated with the good asset management technology;

        - sigma_B: the instantaneous variance associated with the bad asset management technology.

        Based on these parameters, the expected present value of the bank's cash flows from time 0 onwards is computed
        for both the good and the bad technologies, so as to determine with what technology the bank starts.
        """
        self.cash_flows = [x_0]

        self.b = b
        self.r = r

        self.mu_G = mu_G
        self.sigma_G = sigma_G

        self.mu_B = mu_B
        self.sigma_B = sigma_B

        # We compare the expected present value of the bank's cash flows with the good and the bad technology
        if (x_0 / (self.r - self.mu_G) - self.b) >= (x_0 / (self.r - self.mu_B)):

            # If condition is satisfied, the bank first chooses the good asset management technology
            self.technology_choices = ['good']

        else:

            # If condition is not satisfied, the bank first chooses the bad asset management technology
            self.technology_choices = ['bad']

    def generate_cash_flows(self, n_periods=200, dt=0.01, random_seed=None, verbose=0):
        """
        This method allows to simulate the cash flows of the bank.

        It does not return any output but modifies the "cash_flows" and "technology_choices" attribute of the instance
        of the TypicalBank class considered.

        It takes as arguments:

        - n_periods: the number of time increments covered by the simulation (cash flows composed of n_periods values);

        - dt: the length of each time step which is used to simulate a Geometric Brownian motion as a discrete sequence;

        - random_seed: an integer that allows to pre-determine the "state of the world" for the simulation (ie. several
        simulations with the same random seed yield the same output);

        - verbose: determines whether to print or not a message indicating that the attributes have been updated.

        In order to run the simulation, this function relies on the generate_GBM function, imported from utils.py and
        called at each time increment. We cannot generate directly a n_periods-long cash flow series since at any point
        in time, the bank should have the possibility to "shirk", ie. to move from the good to the bad technology.
        """

        # If a random seed is specified, this means that we want the same output each time we call this method
        if random_seed is not None:
            # To do so, we use the provided random seed to generate (the draw is random but we will always obtain the
            # same with a given seed) n_periods random seeds which will be used when calling the generate_GBM function
            np.random.seed(random_seed)
            random_seeds = np.random.randint(1, 1000000, size=n_periods)

        # We iterate to simulate a n_periods-long geometric Brownian motion
        for i in range(n_periods-1):
            # We fetch the current technology choice of the bank from the related instance attribute
            technology = self.technology_choices[-1]

            # We use this technology choice to determine what mu and sigma to use at this time step
            if technology == 'good':
                mu = self.mu_G
                sigma = self.sigma_G
            else:
                mu = self.mu_B
                sigma = self.sigma_B

            # We fetch the current level of cash flows of the bank from the related instance attribute
            x_t = self.cash_flows[-1]

            # Here, we distinguish two cases depending on whether a random seed was specified or not
            if random_seed is not None:

                # We use the n-periods random seeds we have generated to determine the "state of the world" in which
                # the simulation of the geometric Brownian motion at the (i+1)-th time step occurs
                gbm_draw = generate_GBM(mu=mu, sigma=sigma, n=1, dt=dt, x_0=x_t, random_seed=random_seeds[i])[-1]

            else:

                # Here, we do not specify any random seed and let the generate_GBM function run undeterministically
                gbm_draw = generate_GBM(mu=mu, sigma=sigma, n=1, dt=dt, x_0=x_t)[-1]

            # We append the new cash flow level of the bank to its cash_flows attribute
            self.cash_flows.append(gbm_draw)

            # And we determine the next technology choice of the bank based on this cash_flow level
            if (self.cash_flows[-1] / (self.r - self.mu_G) - self.b) > (self.cash_flows[-1] / (self.r - self.mu_B)):
                self.technology_choices.append('good')
            else:
                self.technology_choices.append('bad')

        if verbose:
            print('Cash-flow and technology choice attributes were updated.')

    def plot_cash_flows(self):
        """
        This method allows to rapidly plot the cash flows of the bank.

        It does not require any argument but runs a check to verify that cash flows have been generated beforehand.
        """

        # This check ensures that some cash flows have been generated beforehand
        if len(self.cash_flows) == 1:
            raise Exception("Run a simulation of the bank's cash flows before plotting them.")

        # We simply output a lineplot of the bank's cash flows (the x-axis corresponding to periods)
        plt.plot(self.cash_flows)
        plt.show()

    def has_shirked(self):
        """
        This method allows to check whether a bank has shirked, ie. has chosen the bad technology at some point in time,
        or not.

        It does not require any argument and outputs a boolean which is:

        - True: the bank has shirked, ie. has chosen the bad technology at some point in time;
        - False: the bank has not shirked and kept running with the good technology throughout time.

        NB: Here, we do not run any check to verify for instance whether cash flows have been generated in the first
        place because the first cash flow level, x_0, is determined at the instantiation of the bank and if low enough,
        can potentially induce the bank to choose the bad technology from start.
        """
        return ('bad' in self.technology_choices)


class Economy:

    def __init__(self, b=6, r=1.2, mu_G=1, sigma_G=0.3, mu_B=0.8, sigma_B=0.4, lambda_parameter=2.9, d=1):
        """
        This is the instantiation method for the Economy class.

        It requires as arguments:

        - b: the monitoring cost associated with the good asset management technology;

        - r: the interest rate;

        - mu_G: the instantaneous drift associated with the good asset management technology;

        - mu_B: the instantaneous drift associated with the bad asset management technology;

        - sigma_G: the instantaneous variance associated with the good asset management technology;

        - sigma_B: the instantaneous variance associated with the bad asset management technology;

        - lambda_parameter, which corresponds to lambda in the paper by Decamps, Rochet and Roger and, ie. the parameter
        that determines the liquidation value (lambda * x) of each bank in the economy;

        - d: the amount of (risky) deposits held by each bank in the economy.

        Beside instantiation itself, this method runs several checks on provided parameters to verify that the various
        assumptions posed by authors do hold.
        """

        # Assumption that needs to hold for most computations in the paper (mentioned p. 137)
        if r <= mu_G:
            raise Exception('The interest rate must be strictly greater than the mu_G paramater.')

        # This check and the one that follows ensure that there is a form of hierarchy between good and bad technologies
        if mu_G < mu_B:
            raise Exception('Due to the "hierarchy" between good and the bad technologies, mu_G must be above mu_B.')

        if sigma_B < sigma_G:
            raise Exception('Because the bad technology is more risky, sigma_B must be above sigma_G.')

        # This check verifies a technical assumption made by authors on GBM parameters
        if sigma_G ** 2 >= ((mu_G + mu_B) / 2):
            raise Exception('Technical assumption not satisfied (cf. page 138 of the paper).')

        # Assumption on the lambda parameter
        # Eg., the lower bound implies that closure is always preferable to letting a bank run with the bad technology
        if (lambda_parameter < 1 / (r - mu_B)) or (lambda_parameter > 1 / (r - mu_G)):
            error_message = 'Condition on the lambda parameter is not satisfied. In this case, '
            error_message += f'value must lie between {round(1 / (r - mu_B), 2)} and {round(1 / (r - mu_G), 2)}.'
            raise Exception(error_message)

        # This check and the one below are related to the bank's liabilities (detailed at p. 141)
        if r * d / (r - mu_G) - d <= 0:
            raise Exception('When liquidation takes place, the book value of the bank equity must be positive.')

        if r * d * lambda_parameter >= 1:
            raise Exception('Liquidation should not permit the repayment of all deposits, which would not be risky.')

        self.b = b
        self.r = r

        self.mu_G = mu_G
        self.sigma_G = sigma_G

        self.mu_B = mu_B
        self.sigma_B = sigma_B

        self.lambda_parameter = lambda_parameter

        # This attribute will eventually be "filled" with the output of the simulation
        self.simulation = None

        # This attribute is eventually "filled" within the apply_first_best_closure method
        self.a_G = None

    def get_one_bank(self, x_0=10):
        """
        This method allows to instantiate a bank, using economy-wide parameters.

        It only requires x_0 as an argument, which corresponds to the initial cash flow level of the bank.

        It returns an instance from the TypicalBank class, defined aboveself.
        """
        return TypicalBank(x_0=x_0,
                           b=self.b, r=self.r,
                           mu_G=self.mu_G, sigma_G=self.sigma_G,
                           mu_B=self.mu_B, sigma_B=self.sigma_B)

    def run_simulation(self, n_banks=100,
                       lower_x_0=2, higher_x_0=5,
                       n_periods=200, dt=0.01,
                       fix_random_state=False,
                       inplace=True):
        """

        """
        self.util = [f'cf_{i}' for i in range(n_periods)]

        ids = np.arange(1, n_banks+1)

        all_cash_flows = []
        has_shirkeds = []

        if fix_random_state:
            np.random.seed(0)
            x_0s = np.random.uniform(lower_x_0, higher_x_0, size=len(ids))

            for i, x_0 in zip(ids, x_0s):
                bank = self.get_one_bank(x_0=x_0)
                bank.generate_cash_flows(n_periods=n_periods, dt=dt, random_seed=i)

                all_cash_flows.append(bank.cash_flows)
                has_shirkeds.append(bank.has_shirked())

        else:
            x_0s = np.random.uniform(lower_x_0, higher_x_0, size=len(ids))

            for i, x_0 in zip(ids, x_0s):
                bank = self.get_one_bank(x_0=x_0)
                bank.generate_cash_flows(n_periods=n_periods, dt=dt)

                all_cash_flows.append(bank.cash_flows)
                has_shirkeds.append(bank.has_shirked())

        df = pd.DataFrame(all_cash_flows, columns=[f'cf_{j}' for j in range(len(all_cash_flows[0]))])

        df['bank_id'] = ids
        df['has_shirked'] = has_shirkeds
        df.set_index('bank_id', inplace=True)

        self.nu_G = 1 / (self.r - self.mu_G)
        threshold = self.b / (self.nu_G - self.lambda_parameter)
        df['has_shirked_or_neg_NPV'] = df.apply(lambda row: NPV_check(row, threshold), axis=1)

        if inplace:
            self.simulation = df
        else:
            return df.set_index('bank_id')

    def apply_first_best_closure(self, inplace=True, verbose=1):
        if self.simulation is None:
            raise Exception('You need to first run a simulation before applying first-best closure.')

        self.a_G = (1/2) + (self.mu_G / (self.sigma_G ** 2)) +\
            np.sqrt(((self.mu_G / (self.sigma_G ** 2)) - (1/2)) ** 2 + (2 * self.r) / (self.sigma_G ** 2))

        threshold = (self.b * (self.a_G - 1)) / ((self.nu_G - self.lambda_parameter) * self.a_G)

        self.simulation['first_best_closure'] =\
            self.simulation.apply(lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1)

        if verbose:
            print('Simulation attribute (DataFrame) updated with the first best closure column.')

    def apply_capital_requirements(self, inplace=True, verbose=1):
        if self.a_G is None:
            self.a_G = (1/2) + (self.mu_G / (self.sigma_G ** 2)) +\
                np.sqrt(((self.mu_G / (self.sigma_G ** 2)) - (1/2)) ** 2 + (2 * self.r) / (self.sigma_G ** 2))

        self.nu_B = 1 / (self.r - self.mu_B)

        self.a_B = (1/2) + (self.mu_B / (self.sigma_B ** 2)) +\
            np.sqrt(((self.mu_B / (self.sigma_B ** 2)) - (1/2)) ** 2 + (2 * self.r) / (self.sigma_B ** 2))

        threshold = ((self.a_G - 1) * self.b + self.a_G - self.a_B) / (self.a_G * self.nu_G - self.a_B * self.nu_B)

        self.simulation['capital_requirements_closure'] =\
            self.simulation.apply(lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1)

        if verbose:
            print('Simulation attribute (DataFrame) updated with the "capital_requirements_closure" column.')
