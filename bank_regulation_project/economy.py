"""
This module is the central one in the bank regulation project.

It defines two classes, TypicalBank and Economy, whose methods we use to run simulations.

Formulas, assumptions and economic interpretations used throughout the file are taken from "The three pillars of Basel
II: optimizing the mix", a paper published by Jean-Paul Decamps, Jean-Charles Rochet and Benoît Roger in the "Journal of
Financial Intermediation" in 2002.

Details about how the simulations are conceived can be found in the description of each class and of related methods.
"""

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTS

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from bank_regulation_project.utils import generate_GBM, NPV_check, get_a_exponent, determine_line_color

from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
# DIVERSE

root = 'https://raw.githubusercontent.com/pechouc/bank_regulation_project/main/simulations/files/'

MONTE_CARLO_SIMULATION_PATHS = {
    0: root + 'monte_carlo_output_03_10_200_banks.csv',
    1: root + 'monte_carlo_output_03_11_500_banks.csv',
    2: root + 'monte_carlo_output_03_21_500_banks.csv',
    3: root + 'monte_carlo_output_03_22_500_banks.csv'
}


# ----------------------------------------------------------------------------------------------------------------------
# CONTENT

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
        for i in range(n_periods - 1):
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

    def __init__(self, b, r, mu_G, sigma_G, mu_B, sigma_B, lambda_parameter):
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
            raise Exception('Due to the "hierarchy" between good and bad technologies, mu_G must be above mu_B.')

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
        if r / (r - mu_G) - 1 <= 0:
            raise Exception('When liquidation takes place, the book value of the bank equity must be positive.')

        if r * lambda_parameter >= 1:
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

        # This attribute is eventually "filled" within the initiate_macro_shock method
        self.severe_outcome_mu_G = None

        # These two attributes will eventually be "filled" when analysing the consequences of a macroeconomic shock
        self.first_best_threshold_post_shock = None
        self.capital_requirements_threshold_post_shock = None

    def get_one_bank(self, x_0):
        """
        This method allows to instantiate a bank, using economy-wide parameters.

        It only requires x_0 as an argument, which corresponds to the initial cash flow level of the bank.

        It returns an instance from the TypicalBank class, defined aboveself.
        """
        return TypicalBank(x_0=x_0,
                           b=self.b, r=self.r,
                           mu_G=self.mu_G, sigma_G=self.sigma_G,
                           mu_B=self.mu_B, sigma_B=self.sigma_B)

    def run_first_simulation(self, n_banks=100,
                             lower_x_0=2, upper_x_0=5,
                             n_periods=200, dt=0.01,
                             fix_random_state=False,
                             inplace=True):
        """
        This method is the core simulation method provided by the Economy class.

        It requires several arguments:

        - n_banks: the number of banks to include in the simulation;

        - lower_x_0: the lower bound for the support of the uniform distribution that determines the initial cash flow
        level of each bank;

        - upper_x_0: the upper bound for the support of the uniform distribution that determines the initial cash flow
        level of each bank;

        - n_periods: the number of periods during which one wants to simulate the banks' cash flows;

        - dt: the timestep to be used when generating banks' cash flows using geometric Brownian motions;

        - fix_random_state: boolean that indicates whether to fix or not the random state of the simulation. If set to
        True, the output of the method will be the same through a call to another; if set to False, different calls will
        of the method will yield different outputs;

        - inplace: boolean to indicate whether to store the output in the simulation attribute of the Economy instance
        without returning anything (True) or to return the output instead (False). In the latter case, simulation attri-
        bute is not modified.

        Based on these arguments and using the generate_cash_flows method of TypicalBank instances, this method instan-
        tiate a number of banks, simulates their cash flows and determines whether they have shirked or their assets
        have reached a negative net present value at some point in time.

        It returns a DataFrame:

        - indexed by banks' ID, which range from 1 to n_banks;

        - whose n_periods first columns (called "cf_0", "cf_1", etc) store the cash flow levels of each bank at each
        point in time;

        - with an additional "has_shirked" column which indicates whether the bank has chosen the bad technology at some
        point in time or not;

        - and a last "has_shirked_or_neg_NPV" which stores booleans. True indicates that the bank has either shirked or
        reach a negative net present value at some point in time (which is possible with the good technology when cash
        flows are insufficient to compensate for the monitoring cost).

        NB: This final column is built using the NPV_check function imported from the utils module.
        """

        # This attribute stores the names of columns that will contain banks' cash flows in the output DataFrame
        # It will be reused later on, eg. in apply_first_best_closure and apply_capital_requirements methods
        self.util = [f'cf_{i}' for i in range(n_periods)]

        # This attribute stores the timestep chosen for the simulation and will be reused for post-shock simulations
        self.dt = dt

        # We create the list of bank IDs from 1 to n_banks (a NumPy array to be precise)
        ids = np.arange(1, n_banks + 1)

        # We instantiate void lists that will store banks' cash flows and has_shirked booleans
        all_cash_flows = []
        has_shirkeds = []

        # We distinguish two cases depending on whether we want the output to be the same through different calls
        if fix_random_state:

            # We generate the initial cash flow level of each bank fixing the random seed at 0
            # The x_0s are determined through a continuous uniform distribution of support [lower_x_0; upper_x_0]
            np.random.seed(0)
            x_0s = np.random.uniform(lower_x_0, upper_x_0, size=n_banks)

            # We iterate over bank IDs and the array containing the different initial cash flow levels (same length)
            for i, x_0 in zip(ids, x_0s):
                # We instantiate a bank (from the TypicalBank class) with initial cash flow level x_0
                bank = self.get_one_bank(x_0=x_0)

                # We generate the bank's cash flows which are stored in its cash_flows attribute
                bank.generate_cash_flows(n_periods=n_periods, dt=dt, random_seed=i)

                # We append bank's cash flows and the output of the has_shirked method to related objects
                all_cash_flows.append(bank.cash_flows)
                has_shirkeds.append(bank.has_shirked())

        else:
            # As before, but this time without specifying any random seed, we generate initial cash flow levels
            x_0s = np.random.uniform(lower_x_0, upper_x_0, size=len(ids))

            # We iterate over the array containing the different initial cash flow levels
            for x_0 in x_0s:
                # We instantiate a bank (from the TypicalBank class) with initial cash flow level x_0
                bank = self.get_one_bank(x_0=x_0)

                # We generate the bank's cash flows which are stored in its cash_flows attribute
                bank.generate_cash_flows(n_periods=n_periods, dt=dt)

                # We append bank's cash flows and the output of the has_shirked method to related objects
                all_cash_flows.append(bank.cash_flows)
                has_shirkeds.append(bank.has_shirked())

        # In the end, all_cash_flows is a list of list and we convert this 2-dimensional object into a DataFrame
        # (all_cash_flows contains n_banks lists of n_periods cash flow levels generated as a geometric Brownian motion)
        df = pd.DataFrame(all_cash_flows, columns=self.util)

        # We add columns of interest
        df['bank_id'] = ids
        df['has_shirked'] = has_shirkeds

        # We reindex the DataFrame with banks' IDs
        df.set_index('bank_id', inplace=True)

        # Based on formulas that are detailed in the paper, we compute the positive net present value threshold
        # (If a bank using the good monitoring technology has a cash flow level below this threshold, then the NPV of
        # its assets is non-positive as bank's cash flows cannot compensate for the cost of the monitoring technology)
        self.nu_G = 1 / (self.r - self.mu_G)
        threshold = self.b / (self.nu_G - self.lambda_parameter)

        # We run the check to identify banks that have reached a negative net present value at some point in time
        df['has_shirked_or_neg_NPV'] = df.apply(lambda row: NPV_check(row, threshold), axis=1)

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            self.simulation = df   # simulation attribute of the Economy instance is updated
        else:
            return df   # The attribute is left unchanged and the output is directly returned

    def apply_first_best_closure(self, inplace=True, verbose=1):
        """
        Based on the formula described in Proposition 1 (page 140 of the paper), this method applies the first-best clo-
        sure threshold of the regulator, ie. the threshold which maximizes the option value associated to the irreversi-
        ble closure decision.

        In practice, the first step is to compute this threshold using the parameters of the economy and then, a check
        is run upon each line of the simulation DataFrame to verify, for each bank, whether its cash flows have gone
        below the closure threshold at some point in time.

        It then creates a new column, 'first_best_closure', which takes the value:

        - True, if the bank should have been closed at some point in time based on the first-best threshold;

        - False, if not.

        This method takes two simple arguments:

        - inplace: boolean to indicate whether to store the output in the simulation attribute of the Economy instance
        without returning anything (True) or to return the output instead (False). In the latter case, simulation attri-
        bute is not modified;

        - verbose: determines whether to print or not a message indicating that the attributes have been updated.
        """

        # We first run a check to verify that a simulation has been run and stored in the related attribute beforehand
        if self.simulation is None:
            raise Exception('You need to first run a simulation before applying first-best closure.')

        # We use the formula detailed at page 139 of the paper to compute a_G based on economy parameters
        # In practice, we use the get_a_exponent function that reproduces the formula in utils.py
        self.a_G = get_a_exponent(mu=self.mu_G, sigma=self.sigma_G, r=self.r)

        # We deduce the first-best closure threshold
        threshold = (self.b * (self.a_G - 1)) / ((self.nu_G - self.lambda_parameter) * self.a_G)

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['first_best_closure'] =\
                self.simulation.apply(lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute (DataFrame) updated with the first-best closure column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['first_best_closure'] = df.apply(lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def apply_capital_requirements(self, inplace=True, verbose=1):
        """
        Based on the formula described in Proposition 2 (page 143 of the paper), this method applies the second-best
        closure threshold of the regulator, ie. the threshold associated with the optimal capital ratio under which
        banks have no incentive to shirk.

        As above for the first-best closure, the first step is to compute the capital requirements threshold and a check
        is then run upon each line of the simulation DataFrame to determine whether the bank's cash flows have gone be-
        low the closure threshold at some point in time.

        It then creates a new column, 'capital_requirements_closure', which takes the value:

        - True, if the bank should have been closed based on capital requirements at some point in time;

        - False, if not.

        This method takes two simple arguments:

        - inplace: boolean to indicate whether to store the output in the simulation attribute of the Economy instance
        without returning anything (True) or to return the output instead (False). In the latter case, simulation attri-
        bute is not modified;

        - verbose: determines whether to print or not a message indicating that the attributes have been updated.
        """

        # We first run a check to verify that a simulation has been run and stored in the related attribute beforehand
        if self.simulation is None:
            raise Exception('You need to first run a simulation before applying capital requirements.')

        # In case the apply_first_best_closure method has not been run before calling the one considered here,
        # we need to recompute a_G based on economy parameters and thanks to the formula at page 139 of the paper
        if self.a_G is None:
            self.a_G = get_a_exponent(mu=self.mu_G, sigma=self.sigma_G, r=self.r)

        # We compute two components of the final formula related to the bad asset monitoring technology (1/2)
        self.nu_B = 1 / (self.r - self.mu_B)

        # We compute two components of the final formula related to the bad asset monitoring technology (2/2)
        self.a_B = get_a_exponent(mu=self.mu_B, sigma=self.sigma_B, r=self.r)

        # Again based on Proposition 2 of the paper (page 143), we verify capital requirements are needed in our case
        if self.b <= (self.r * (self.a_G * self.nu_G - self.a_B * self.nu_B) - (self.a_G - self.a_B)) / (self.a_G - 1):
            raise Exception(
                'With the considered parameters, capital requirements regulation is not needed.'
                + ' '
                + 'This happens when the monitoring cost is not "large enough" - See Proposition 2 (page 143).'
            )

        # We deduce from previous computations the second-best / capital requirements closure threshold
        threshold = ((self.a_G - 1) * self.b + self.a_G - self.a_B) / (self.a_G * self.nu_G - self.a_B * self.nu_B)

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['capital_requirements_closure'] =\
                self.simulation.apply(lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute (DataFrame) updated with the second-best closure column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['capital_requirements_closure'] = df.apply(
                lambda row: (row.loc[self.util] <= threshold).sum() > 0, axis=1
            )

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def initiate_macro_shock(self,
                             severe_outcome_mu_G, severe_outcome_sigma_G,
                             severe_outcome_mu_B, severe_outcome_sigma_B,
                             light_outcome_mu_G, light_outcome_sigma_G,
                             light_outcome_mu_B, light_outcome_sigma_B,
                             severe_outcome_proba=0.2,
                             verbose=1):
        """
        This function allows to initiate the analysis and simulation of a macroeconomic shock and its effect on bank
        regulation challenges or policies. It encompasses three main objectives:

        - running a variety of checks on the parameters of the macroeconomic shock to simulate;

        - storing these as attributes of the Economy instance;

        - running a few computations which are used in the following methods to simulate and analyse the shock.

        It requires several arguments, that we now detail.

        As a macroeconomic shock leads the regulator to make two different assumptions about its consequences on banks'
        future profitability - the two scenarios respectively corresponding to a "severe" and a "light" outcome -, we
        need for each of the two outcomes:

        - the instantaneous drift of the geometric Brownian motion associated with the good asset monitoring technology.
        It is given in the two cases by the severe_outcome_mu_G and light_outcome_mu_G arguments;

        - the instantaneous drift of the geometric Brownian motion associated with the bad asset monitoring technology.
        It is given in the two cases by the severe_outcome_mu_B and light_outcome_mu_B arguments;

        - the instantaneous variance of the geometric Brownian motion associated with the good technology. It is given
        in the two cases by the severe_outcome_sigma_G and light_outcome_sigma_G arguments;

        - the instantaneous variance of the geometric Brownian motion associated with the bad technology. It is given in
        the two cases by the severe_outcome_sigma_B and light_outcome_sigma_B arguments.

        Besides, the method requires two additional arguments:

        - severe_outcome_proba, which corresponds to the probability (float must thus be comprised between 0 and 1) that
        the severe outcome realizes. It is assumed to be properly evaluated by the regulator;

        - verbose, which is equal to 1 by default, indicates whether to print a confirmation of the shock initiation.
        """

        # We check that severe_outcome_proba corresponds to a well-defined probability
        if severe_outcome_proba > 1 or severe_outcome_proba < 0:
            raise Exception('The probability of the severe outcome must be comprised between 0 and 1.')

        # We first run a check to verify that a simulation has been run and stored in the related attribute beforehand
        if self.simulation is None:
            raise Exception('You need to run a first simulation before initiating a macroeconomic shock.')

        # Assumption that needs to hold for most computations in the paper (mentioned p. 137)
        if self.r <= severe_outcome_mu_G or self.r <= light_outcome_mu_G:
            raise Exception('The interest rate must be strictly greater than the mu_G paramater in both outcomes.')

        # This check and the one that follows ensure that there is a form of hierarchy between good and bad technologies
        if severe_outcome_mu_G < severe_outcome_mu_B or light_outcome_mu_G < light_outcome_mu_B:
            raise Exception(
                'Due to the "hierarchy" between good and bad technologies, mu_G must be above mu_B in both outcomes.'
            )

        if severe_outcome_sigma_G > severe_outcome_sigma_B or light_outcome_sigma_G > light_outcome_sigma_B:
            raise Exception(
                'Because the bad technology is more risky, sigma_B must be above or equal to sigma_G in both outcomes.'
            )

        # This check verifies, in both severe and light outcomes, a technical assumption on GBM parameters
        if (
            severe_outcome_sigma_G ** 2 >= ((severe_outcome_mu_G + severe_outcome_mu_B) / 2)
            or light_outcome_sigma_G ** 2 >= ((light_outcome_mu_G + light_outcome_mu_B) / 2)
        ):
            raise Exception('Technical assumption must be satisfied in both outcomes (cf. page 138 of the paper).')

        # Assumption on the lambda parameter - Severe outcome
        severe_outcome_nu_G = 1 / (self.r - severe_outcome_mu_G)
        severe_outcome_nu_B = 1 / (self.r - severe_outcome_mu_B)
        if self.lambda_parameter < severe_outcome_nu_B or self.lambda_parameter > severe_outcome_nu_G:
            error = 'Condition on the lambda parameter is not satisfied in the severe outcome. In this case, '
            error += f'value must lie between {round(severe_outcome_nu_B, 2)} and {round(severe_outcome_nu_G, 2)}.'
            raise Exception(error)

        # Assumption on the lambda parameter - Light outcome
        light_outcome_nu_G = 1 / (self.r - light_outcome_mu_G)
        light_outcome_nu_B = 1 / (self.r - light_outcome_mu_B)
        if self.lambda_parameter < light_outcome_nu_B or self.lambda_parameter > light_outcome_nu_G:
            error = 'Condition on the lambda parameter is not satisfied in the light outcome. In this case, '
            error += f'value must lie between {round(light_outcome_nu_B, 2)} and {round(light_outcome_nu_G, 2)}.'
            raise Exception(error)

        # The following two checks are related to the bank's liabilities (detailed at p. 141)
        if self.r / (self.r - severe_outcome_mu_G) - 1 <= 0:
            raise Exception(
                'Severe outcome - When liquidation takes place, the book value of the bank equity must be positive.'
            )

        if self.r / (self.r - light_outcome_mu_G) - 1 <= 0:
            raise Exception(
                'Light outcome - When liquidation takes place, the book value of the bank equity must be positive.'
            )

        # Should be uncommented and completed if we decide to let the interest rate vary based on the realized outcome
        # if r * lambda_parameter >= 1:
        #     raise Exception(
        #         'Severe outcome - Liquidation cannot permit the repayment of all deposits, which would not be risky.'
        #     )

        # if r * lambda_parameter >= 1:
        #     raise Exception(
        #         'Light outcome - Liquidation cannot permit the repayment of all deposits, which would not be risky.'
        #     )

        # We store the probability of the most severe macroeconomic shock among the attributes of the Economy instance
        self.severe_outcome_proba = severe_outcome_proba

        # We add to the object attributes the parameters of the good and bad technology motions in case of severe shock
        self.severe_outcome_mu_G = severe_outcome_mu_G
        self.severe_outcome_sigma_G = severe_outcome_sigma_G
        self.severe_outcome_mu_B = severe_outcome_mu_B
        self.severe_outcome_sigma_B = severe_outcome_sigma_B

        # We add to the object attributes the parameters of the good and bad technology motions in case of light shock
        self.light_outcome_mu_G = light_outcome_mu_G
        self.light_outcome_sigma_G = light_outcome_sigma_G
        self.light_outcome_mu_B = light_outcome_mu_B
        self.light_outcome_sigma_B = light_outcome_sigma_B

        # We store as attributes several scalars defined in the paper, which will prove useful later on
        # First, in the case of a severe outcome
        self.severe_outcome_nu_G = severe_outcome_nu_G
        self.severe_outcome_a_G = get_a_exponent(mu=severe_outcome_mu_G, sigma=severe_outcome_sigma_G, r=self.r)
        self.severe_outcome_nu_B = severe_outcome_nu_B
        self.severe_outcome_a_B = get_a_exponent(mu=severe_outcome_mu_B, sigma=severe_outcome_sigma_B, r=self.r)
        # Then, in the case of a light outcome
        self.light_outcome_nu_G = light_outcome_nu_G
        self.light_outcome_a_G = get_a_exponent(mu=light_outcome_mu_G, sigma=light_outcome_sigma_G, r=self.r)
        self.light_outcome_nu_B = light_outcome_nu_B
        self.light_outcome_a_B = get_a_exponent(mu=light_outcome_mu_B, sigma=light_outcome_sigma_B, r=self.r)

        if verbose:
            print('Macroeconomic shock initiated successfully.')

    def apply_first_best_closure_under_shock(self, strategy='balanced', inplace=True, verbose=1):
        """
        This method allows to compute and apply the first-best closure threshold of the regulator, ie. the one which
        maximizes the continuation value of banks in the economy, updated with the parameters of the macroeconomic
        shock (that has to be initiated in the first place).

        As before, it is based on the formula described in Proposition 1 (page 140 of the paper).

        In practice, the first step is to compute this threshold for both the severe outcome and the light outcome that
        may both actually realize with some probability. Then, depending on the "strategy" passed as argument (more on
        this below), an aggregated threshold is determined. Eventually, a check is run on the last cash-flow level of
        each bank to verify whether it is above or below the new closure threshold.

        The method then creates a new column, 'first_best_closure_under_shock', which takes the value:

        - True, if the bank should have been closed based on the new threshold at the time of the shock;

        - False, if not.

        It requires three arguments:

        - strategy: this argument, that can take either "balanced" or "prudent" as value, indicates what method to use
        when aggregating the severe outcome and light outcome thresholds:

            - "balanced" would correspond to a risk-neutral regulator and in this case, the threshold applied is the
            mean of the severe and light outcome thresholds,

            - "prudent" would instead correspond to an extremely risk-averse regulator, in which case the severe outcome
            threshold is directly applied;

        - inplace: this boolean indicates whether to directly update the simulation attribute of the Economy instance
        (inplace=True) or to return a copy of the simulation DataFrame with the new column (inplace=False). It is set to
        True by default;

        - verbose, which is equal to 1 by default, indicates whether to print a confirmation of the application of the
        new closure threshold.
        """

        # We run a check to verify that the strategy argument being passed corresponds to one of the two possibilities
        if strategy not in ['balanced', 'prudent']:
            raise Exception('The strategy of the regulator can either be "balanced" or "prudent".')

        # We run a check to verify that a first simulation has been run
        if self.simulation is None:
            raise Exception('You need to run a first simulation before analysing a macroeconomic shock.')

        # We then run a check to verify that a macroeconomic shock has been initiated
        if self.severe_outcome_mu_G is None:
            raise Exception('You need to initiate a macroeconomic shock before you can apply a threshold under shock.')

        # We first need to compute the first-best closure threshold under the severe outcome
        severe_outcome_threshold = (self.b * (self.severe_outcome_a_G - 1)) / \
            ((self.severe_outcome_nu_G - self.lambda_parameter) * self.severe_outcome_a_G)

        # We then compute the first-best closure threshold under the light outcome
        light_outcome_threshold = (self.b * (self.light_outcome_a_G - 1)) / \
            ((self.light_outcome_nu_G - self.lambda_parameter) * self.light_outcome_a_G)

        # We compute the balanced closure threshold of the regulator, which does not know what outcome is realized
        self.first_best_threshold_post_shock = self.severe_outcome_proba * severe_outcome_threshold + \
            (1 - self.severe_outcome_proba) * light_outcome_threshold

        # We determine the threshold eventually applied by the regulator depending on the selected strategy
        if strategy == 'balanced':
            threshold = self.first_best_threshold_post_shock
        else:
            threshold = severe_outcome_threshold

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['first_best_closure_under_shock'] =\
                self.simulation[self.util[-1]].map(lambda x: x <= threshold)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute updated with the first-best closure under shock column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['first_best_closure_under_shock'] = df[self.util[-1]].map(lambda x: x <= threshold)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def apply_capital_requirements_under_shock(self, strategy='balanced', inplace=True, verbose=1):
        """
        This method allows to compute and apply the second-best closure threshold of the regulator, ie. the one which is
        assimilated with a capital requirements ratio in the paper, updated with the parameters of the macroeconomic
        shock (that has to be initiated in the first place).

        As before, it is based on the formula described in Proposition 2 (page 143 of the paper).

        In practice, the first step is to compute this threshold for both the severe outcome and the light outcome that
        may both actually realize with some probability. Then, depending on the "strategy" passed as argument (more on
        this below), an aggregated threshold is determined. Eventually, a check is run on the last cash-flow level of
        each bank to verify whether it is above or below the new closure threshold.

        The method then creates a new column, 'capital_requirements_closure_under_shock', which takes the value:

        - True, if the bank should have been closed based on the new threshold at the time of the shock;

        - False, if not.

        It requires three arguments:

        - strategy ("balanced" by default): this argument, that can take either "balanced" or "prudent" as value, indi-
        cates what method to use when aggregating the severe outcome and light outcome thresholds:

            - "balanced" would correspond to a risk-neutral regulator and in this case, the threshold applied is the
            mean of the severe and light outcome thresholds,

            - "prudent" would instead correspond to an extremely risk-averse regulator, in which case the severe outcome
            threshold is directly applied.

        - inplace: this boolean indicates whether to directly update the simulation attribute of the Economy instance
        (inplace=True) or to return a copy of the simulation DataFrame with the new column (inplace=False). It is set to
        True by default;

        - verbose, which is equal to 1 by default, indicates whether to print a confirmation of the application of the
        new closure threshold.
        """

        # We run a check to verify that the strategy argument being passed corresponds to one of the two possibilities
        if strategy not in ['balanced', 'prudent']:
            raise Exception('The strategy of the regulator can either be "balanced" or "prudent" (cf. documentation).')

        # We run a check to verify that a first simulation has been run
        if self.simulation is None:
            raise Exception('You need to run a first simulation before analysing a macroeconomic shock.')

        # We then run a check to verify that a macroeconomic shock has been initiated
        if self.severe_outcome_mu_G is None:
            raise Exception('You need to initiate a macroeconomic shock before you can apply a threshold under shock.')

        # We first compute the second-best / capital requirements closure threshold under the severe outcome
        severe_outcome_threshold = (
            ((self.severe_outcome_a_G - 1) * self.b + self.severe_outcome_a_G - self.severe_outcome_a_B) /
            (self.severe_outcome_a_G * self.severe_outcome_nu_G - self.severe_outcome_a_B * self.severe_outcome_nu_B)
        )

        # We then compute the second-best / capital requirements closure threshold under the light outcome
        light_outcome_threshold = (
            ((self.light_outcome_a_G - 1) * self.b + self.light_outcome_a_G - self.light_outcome_a_B) /
            (self.light_outcome_a_G * self.light_outcome_nu_G - self.light_outcome_a_B * self.light_outcome_nu_B)
        )

        # We compute the balanced closure threshold of the regulator, which does not know what outcome is realized
        self.capital_requirements_threshold_post_shock = self.severe_outcome_proba * severe_outcome_threshold + \
            (1 - self.severe_outcome_proba) * light_outcome_threshold

        # We determine the threshold eventually applied by the regulator depending on the selected strategy
        if strategy == 'balanced':
            threshold = self.capital_requirements_threshold_post_shock
        else:
            threshold = severe_outcome_threshold

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['capital_requirements_closure_under_shock'] =\
                self.simulation[self.util[-1]].map(lambda x: x <= threshold)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute updated with the capital requirements closure under shock column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['capital_requirements_closure_under_shock'] = df[self.util[-1]].map(lambda x: x <= threshold)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def simulate_macro_shock(self,
                             n_periods=200,
                             fix_random_state=False, selected_outcome=None,
                             inplace=True):
        """
        This is the second simulation method provided by the Economy class.

        It builds upon the first simulation of cash flows to pursue each bank's sequence after a macroeconomic shock.
        A first simulation must have been run and a macroeconomic shock must have been initiated beforehand, respective-
        ly with the run_first_simulation and initiate_macro_shock methods.

        It requires several arguments:

        - n_periods, the number of periods during which one wants to simulate the banks' cash flows after the shock;

        - fix_random_state, a boolean which indicates whether to fix the random state of the simulation. If it is set to
        True, the output will be the same from a run to another while the output is allowed to vary if False is passed;

        - selected_outcome, this argument is required if and only if one wants to fix the random state for the simula-
        tion. Either "severe" or "light" can be passed, determining what set of parameters will be used to generate
        banks' cash flows under the shock. If a selected_outcome is specified while passing fix_random_state=False, this
        will have no impact on the simulation and the realized outcome will be determined randomly.

        - inplace, set to True by default, specifies whether to directly update the simulation attribute of the Economy
        instance (inplace=True) or to return a copy of the simulation DataFrame with the new column (inplace=False).

        It either updates the simulation attribute of the Economy instance or returns a copy with several new columns:

        - n_periods columns (for instance denominated by "cf_200", "cf_399", etc), which store the cash flow levels of
        each bank at each point in time after the macroeconomic shock;

        - a "has_shirked_post_shock" column which indicates whether the bank has chosen the bad technology at some
        point in time after the macroeconomic shock or not;

        - and a last "has_shirked_or_neg_NPV_post_shock" which stores booleans. True indicates that the bank has either
        shirked or reached a negative net present value at some point in time after the shock (which is possible with
        the good technology when cash flows are insufficient to compensate for the monitoring cost).

        NB: This final column is built using the NPV_check function imported from the utils module.
        """

        # We first run a check to verify that a  macroeconomic shock has been initiated
        if self.severe_outcome_mu_G is None:
            raise Exception('You need to initiate a macroeconomic shock before you can simulate it.')

        # This attribute stores the names of columns that will contain banks' cash flows generated under a macroeconomic
        # shock in the output DataFrame (for example, from "cf_200" to "cf_399")
        self.util_bis = [f'cf_{i+len(self.util)}' for i in range(n_periods)]

        # We fetch the list of bank IDs from 1 to n_banks (a NumPy array to be precise)
        ids = self.simulation.index.values

        # We instantiate void lists that will store cash flows and has_shirked booleans under post-shock conditions
        all_cash_flows = []
        has_shirkeds = []

        # x_0's are not generated randomly here; they correspond to the t=(n_periods-1) cash flow level of each bank
        x_0s = self.simulation[self.util[-1]].values

        # We distinguish two processes for the simulation depending on the fix_random_state argument
        # And we first focus on the case where random state is fixed for the output to be the same from a run to another
        if fix_random_state:

            # We run a check to verify that an outcome (either "severe" or "light") has been specified
            if selected_outcome is None:
                raise Exception('If you want to fix the random state, you need to specify what outcome gets realised.')

            # We store the realized outcome in a dedicated attribute of the Economy instance
            self.realized_outcome = selected_outcome

            # Based on the selected outcome argument, we determine what parameters to use to generate banks' cash flows
            if selected_outcome == 'severe':
                # In this case, the severe outcome is realised
                mu_G = self.severe_outcome_mu_G
                sigma_G = self.severe_outcome_sigma_G
                mu_B = self.severe_outcome_mu_B
                sigma_B = self.severe_outcome_sigma_B

                # We set a coefficient that will be used to fix the random seeds when generating banks' cash flows
                # Indeed, we do not want the same random state to be used in both "severe" and "light" outcomes
                coeff = 2

            elif selected_outcome == 'light':
                # In this case, the light outcome is realised
                mu_G = self.light_outcome_mu_G
                sigma_G = self.light_outcome_sigma_G
                mu_B = self.light_outcome_mu_B
                sigma_B = self.light_outcome_sigma_B

                # We set the random seed coefficient to a different value than in the "severe" case
                coeff = 3

            # We iterate over ids and the array containing the different initial cash flow levels (same length)
            for i, x_0 in zip(ids, x_0s):
                # We instantiate a bank (from the TypicalBank class) with initial cash flow level x_0
                bank = TypicalBank(x_0=x_0,
                                   b=self.b, r=self.r,
                                   mu_G=mu_G, sigma_G=sigma_G,
                                   mu_B=mu_B, sigma_B=sigma_B)

                # We generate the bank's cash flows which are stored in its cash_flows attribute
                bank.generate_cash_flows(n_periods=(n_periods + 1), dt=self.dt, random_seed=(i + coeff * len(ids)))

                # We append bank's cash flows and the output of the has_shirked method to related objects
                all_cash_flows.append(bank.cash_flows[1:])
                has_shirkeds.append(bank.has_shirked())

        # In this second case, the random state is not fixed the simulation is fully random
        else:

            # We now need to determine what outcome is realised, either the "severe" or the "light" one
            random_draw = np.random.rand()

            if random_draw < self.severe_outcome_proba:
                # We store the realized outcome in a dedicated attribute of the Economy instance
                self.realized_outcome = 'severe'

                # In this case, the severe outcome is realised
                mu_G = self.severe_outcome_mu_G
                sigma_G = self.severe_outcome_sigma_G
                mu_B = self.severe_outcome_mu_B
                sigma_B = self.severe_outcome_sigma_B

            elif random_draw >= self.severe_outcome_proba:
                # We store the realized outcome in a dedicated attribute of the Economy instance
                self.realized_outcome = 'light'

                # In this case, the light outcome is realised
                mu_G = self.light_outcome_mu_G
                sigma_G = self.light_outcome_sigma_G
                mu_B = self.light_outcome_mu_B
                sigma_B = self.light_outcome_sigma_B

            # We iterate over the array containing the different initial cash flow levels
            for x_0 in x_0s:
                # We instantiate a bank (from the TypicalBank class) with initial cash flow level x_0
                bank = TypicalBank(x_0=x_0,
                                   b=self.b, r=self.r,
                                   mu_G=mu_G, sigma_G=sigma_G,
                                   mu_B=mu_B, sigma_B=sigma_B)

                # We generate the bank's cash flows which are stored in its cash_flows attribute
                bank.generate_cash_flows(n_periods=(n_periods + 1), dt=self.dt)

                # We append bank's cash flows and the output of the has_shirked method to related objects
                all_cash_flows.append(bank.cash_flows[1:])
                has_shirkeds.append(bank.has_shirked())

        # In the end, all_cash_flows is a list of list and we convert this 2-dimensional object into a DataFrame
        # (all_cash_flows contains n_banks lists of n_periods cash flow levels generated as a geometric Brownian motion)
        df = pd.DataFrame(all_cash_flows, columns=self.util_bis)

        # We add columns of interest, the second one giving whether the bank has chosen the bad technology at some point
        # in time during the second simulation, ie. under macroeconomic shock conditions
        df['bank_id'] = ids
        df['has_shirked_post_shock'] = has_shirkeds

        # We index the DataFrame by bank IDs
        df.set_index('bank_id', inplace=True)

        # We compute the positive net present value threshold under the new motion parameters in the economy
        # (If a bank using the good monitoring technology has a cash flow level below this threshold, then the NPV of
        # its assets is non-positive as bank's cash flows cannot compensate for the cost of the monitoring technology)
        nu_G = 1 / (self.r - mu_G)
        threshold = self.b / (nu_G - self.lambda_parameter)

        # We run a check to identify banks that have reached a negative net present value at some point under the shock
        df['has_shirked_or_neg_NPV_post_shock'] = df.apply(
            lambda row: NPV_check(
                row=row, threshold=threshold,
                under_macro_shock=True, column_indices=self.util_bis
            ),
            axis=1
        )

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            if 'has_shirked_post_shock' in self.simulation.columns:
                self.simulation.drop(
                    columns=(self.util_bis + ['has_shirked_post_shock', 'has_shirked_or_neg_NPV_post_shock']),
                    inplace=True
                )

            # simulation attribute of the Economy instance is updated
            self.simulation = pd.concat([self.simulation, df], axis=1)
        else:
            return df   # The attribute is left unchanged and the output is directly returned

    def apply_first_best_closure_post_shock(self, inplace=True, verbose=1):
        """
        This method applies to banks' post-shock cash-flow levels the first-best closure threshold of the regulator,
        updated with the parameters of the macroeconomic shock.

        Assumingly, the regulator uses, in the n_periods after the shock, the balanced first-best closure threshold that
        has been computed and stored in the attributes of the Economy instance within the apply_first_best_closure_-
        under_shock method (which therefore has to be run in the first place).

        In this method, the first step is thus to fetch the relevant threshold from pre-stored attributes. A check is
        then run on each bank's post-shock cash-flow levels to verify whether they have gone below the closure threshold
        at some point in time.

        It then creates a new column, 'first_best_closure_post_shock', which takes the value:

        - True, if the bank should have been closed at some point in time after the shock based on the new threshold;

        - False, if not.

        This method takes two simple arguments:

        - inplace: boolean to indicate whether to store the output in the simulation attribute of the Economy instance
        without returning anything (True) or to return the output instead (False). In the latter case, simulation attri-
        bute is not modified;

        - verbose: determines whether to print or not a message indicating that the attributes have been updated, as
        well as the threshold actually applied.
        """

        # We first run a check to verify that the simulation attribute of the Economy instance contains cash flows
        # simulated under macroeconomic shock conditions thanks to the simulate_macro_shock method
        if 'has_shirked_post_shock' not in self.simulation.columns:
            raise Exception('This method requires to have simulated a macroeconomic shock with inplace=True.')

        # We then run a check to verify that the first-best closure threshold under shock has been computed and stored
        if self.first_best_threshold_post_shock is None:
            raise Exception('This method requires to first run the apply_first_best_closure_under_shock method.')

        # We fetch the threshold to apply from the attributes of the Economy instance
        threshold = self.first_best_threshold_post_shock

        # If verbose=1 was passed, we print the threshold being applied
        if verbose:
            print(f'Threshold applied is: {round(threshold, 2)}')

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['first_best_closure_post_shock'] =\
                self.simulation.apply(lambda row: (row.loc[self.util_bis] <= threshold).sum() > 0, axis=1)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute (DataFrame) updated with the post-shock first-best closure column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['first_best_closure_post_shock'] = df.apply(
                lambda row: (row.loc[self.util_bis] <= threshold).sum() > 0, axis=1
            )

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def apply_capital_requirements_post_shock(self, inplace=True, verbose=1):
        """
        This method applies to banks' post-shock cash-flow levels the second-best or capital requirements closure
        threshold of the regulator, updated with the parameters of the macroeconomic shock.

        Assumingly, the regulator uses, in the n_periods after the shock, the balanced capital requirements closure
        threshold that has been computed and stored in the attributes of the Economy instance within the apply_capital_-
        requirements_under_shock method (which therefore has to be run in the first place).

        In this method, the first step is thus to fetch the relevant threshold from pre-stored attributes. A check is
        then run on each bank's post-shock cash-flow levels to verify whether they have gone below the closure threshold
        at some point in time.

        It then creates a new column, 'capital_requirements_closure_post_shock', which takes the value:

        - True, if the bank should have been closed at some point in time after the shock based on the new threshold;

        - False, if not.

        This method takes two simple arguments:

        - inplace: boolean to indicate whether to store the output in the simulation attribute of the Economy instance
        without returning anything (True) or to return the output instead (False). In the latter case, simulation attri-
        bute is not modified;

        - verbose: determines whether to print or not a message indicating that the attributes have been updated, as
        well as the threshold actually applied.
        """

        # We first run a check to verify that the simulation attribute of the Economy instance contains cash flows
        # simulated under macroeconomic shock conditions thanks to the simulate_macro_shock method
        if 'has_shirked_post_shock' not in self.simulation.columns:
            raise Exception('This method requires to have simulated a macroeconomic shock with inplace=True.')

        # We then run a check to verify that the second-best closure threshold under shock has been computed and stored
        if self.capital_requirements_threshold_post_shock is None:
            raise Exception('This method requires to first run the apply_capital_requirements_under_shock method.')

        # We fetch the threshold to apply from the attributes of the Economy instance
        threshold = self.capital_requirements_threshold_post_shock

        # If verbose=1 was passed, we print the threshold being applied
        if verbose:
            print(f'Threshold applied is: {round(threshold, 2)}')

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # simulation attribute of the Economy instance is updated
            self.simulation['capital_requirements_closure_post_shock'] =\
                self.simulation.apply(lambda row: (row.loc[self.util_bis] <= threshold).sum() > 0, axis=1)

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute (DataFrame) updated with the post-shock second-best closure column.')

        else:
            # The attribute is left unchanged and the output is directly returned
            df = self.simulation.copy()
            df['capital_requirements_closure_post_shock'] = df.apply(
                lambda row: (row.loc[self.util_bis] <= threshold).sum() > 0, axis=1
            )

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('Simulation attribute was left unchanged (inplace=False was passed).')

            return df

    def plot_simulation(self, n_lines, plot_shock=False):
        """
        This function allows to plot the output of pre-shock and post-shock simulations of banks' cash flows.

        Based on the results stored in the simulation attribute of the Economy instance, it returns a simple line plot
        of the cash flow sequences of a sub-sample of banks, chosen randomly. The color of the line is determined by the
        bank's choice of technology before or after the macroeconomic shock.

        This method requires two arguments:

        - n_lines (no default value), which determines how many cash flow sequences one wants to visualize. From there,
        the method will draw without replacement n_lines banks from the simulation and display their results;

        - plot_shock is a boolean that indicates whether to plot only cash flows before the macroeconomic shock or both
        pre-shock and post-shock sequences. Indeed, the form of the graph (as described below) differs beween these two
        cases. Naturally, to pass plot_shock=True, it is necessary to simulate a macroeconomic shock in the first place.
        """
        # We run a check to verify that a first simulation has been run
        if self.simulation is None:
            raise Exception('You need to run a first simulation before plotting its results.')

        indices = np.random.choice(self.simulation.index, n_lines, replace=False)

        df = self.simulation.loc[indices, :].copy()

        plt.figure(figsize=(20, 12))

        # We distinguish two cases depending on whether we want to plot banks' post-shock cash flows
        # In this first case, we simply plot pre-shock cash flows
        if not plot_shock:
            # We build the legend of the graph
            legend_elements = [
                Line2D([0], [0], color='darkblue', label='Has not shirked'),
                Line2D([0], [0], color='darkred', label='Has  shirked')
            ]

            # Based on whether the bank has shirked or not, we determine what color to use for this bank's cash flows
            colors = df['has_shirked'].map(lambda x: 'darkred' if x else 'darkblue').values

            # We iterate over each bank's cash-flow sequence and over the list of colors
            for y, color in zip(df[self.util].values, colors):
                # We plot each bank's cash-flow sequence
                plt.plot(np.arange(len(y)), y, color=color)

        # In this second case, we plot pre-shock and post-shock cash flows
        else:
            # We run a check to verify that a macroeconomic shock has been simulated
            if 'has_shirked_post_shock' not in self.simulation.columns:
                raise Exception('This method requires to have simulated a macroeconomic shock with inplace=True.')

            # We build the legend of the graph, with two additional fields compared with the previous case
            legend_elements = [
                Line2D([0], [0], color='darkblue', label='Has not shirked'),
                Line2D([0], [0], color='darkred', label='Has  shirked before the shock'),
                Line2D([0], [0], color='orange', label='Has  shirked after the shock'),
                Line2D([0], [0], color='darkgreen', label='Macroeconomic shock')
            ]

            # Relying on the determine_line_color function imported from the utils module and based on whether the bank
            # has shirked before or after the macroeconomic shock, we determine what color for this bank's cash flows
            colors = df[['has_shirked', 'has_shirked_post_shock']].apply(
                determine_line_color,
                axis=1
            ).values

            # We iterate over each bank's cash-flow sequence and over the list of colors
            for y, color in zip(df[self.util + self.util_bis].values, colors):
                # We plot each bank's cash-flow sequence
                plt.plot(np.arange(len(y)), y, color=color)

            # We add a vertical line indicating at what point in time the macroeconomic shock occurred
            plt.axvline(x=len(self.util), color='darkgreen')

        # We add the legend to the graph
        plt.legend(handles=legend_elements, loc='best', prop={'size': 14})

        # We return the graph
        plt.show()

    def run_monte_carlo_simulation(self, n_trials=200, n_banks=100, inplace=True, verbose=1):
        '''
        This is the third and final simulation method provided by the Economy class.

        This method allows to run Monte-Carlo simulations of the model, in the sense that a large number of Economy in-
        stances are created, so as to run multiple random simulations and draw aggregate results from there.

        As such, this method requires a macroeconomic shock to have been initiated beforehand as the same "normal
        state", severe outcome and light outcome parameters will be used throughout the different trials.

        Four arguments are necessary:

        - n_trials (200 by default), which determines how many Economy instances are created and therefore, how many si-
        mulations will be run;

        - n_banks (100 by default), which determines how many banks will be simulated in each Economy instance;

        - inplace (True by default), which indicates whether to store the output in the monte_carlo_simulation attribute
        of the Economy instance without returning anything (True) or to return the output instead (False);

        - verbose (1 by default), which determines whether to print or not a message indicating that the attributes have
        been updated, as well as the threshold actually applied.

        The output of this method is a DataFrame which contains one line for each trial and the following columns:

        - n_have_shirked: number of banks having shirked before the macroeconomic shock;

        - n_have_shirked_or_neg_NPV: number of banks having shirked or reached a negative net present value before the
        macroeconomic shock;

        - n_first_best_closures: number of banks that should be closed based on the first-best closure threshold of the
        regulator before the macroeconomic shock;

        - n_capital_requirements_closure: number of banks that should be closed based on the second-best or capital re-
        quirements closure threshold of the regulator before the macroeconomic shock;

        - n_first_best_balanced_closures_under_shock: number of previously sound banks (in the sense that they were not
        closed before the macroeconomic shock based on the first-best closure of the regulator) that should be closed
        based on their cash flow level at the moment of the macroeconomic shock and on the balanced first-best closure
        threshold of the regulator;

        - n_first_best_prudent_closures_under_shock: number of previously sound banks that should be closed based on
        their cash flow level at the moment of the macroeconomic shock and on the prudent first-best closure threshold
        of the regulator;

        - n_capital_requirements_balanced_closures_under_shock: number of previously sound banks (this time referring to
        the capital requirements closure threshold of the regulator before the shock) that should be closed based on
        their cash flow level at the moment of the macroeconomic shock and on the balanced capital requirements closure
        threshold of the regulator;

        - n_capital_requirements_prudent_closures_under_shock: number of previously sound banks that should be closed
        based on their cash flow level at the moment of the macroeconomic shock and on the prudent capital requirements
        closure threshold of the regulator;

        - realized_outcome: stores the outcome (either "light" or "severe") that randomly realized when simulating the
        macroeconomic shock;

        - n_have_shirked_post_shock: number of banks, among those that had not shirked before the macroeconomic shock,
        that have shirked after the shock;

        - n_have_shirked_or_neg_NPV_post_shock: number of banks, among those that had neither shirked nor reached a ne-
        gative NPV before the macroeconomic shock, that have either shirked or reached a negative NPV after the shock;

        - n_first_best_closures_post_shock: number of previously sound banks (not closed based on the first-best closure
        threshold of the regulator before the shock) that should be closed after the shock, applying the new balanced
        first-best closure threshold;

        - n_capital_requirements_closures_post_shock: number of previously sound banks (not closed based on the capital
        requirements closure threshold of the regulator before the shock) that should be closed after the shock, apply-
        ing the new balanced capital requirements closure threshold.
        '''

        # We first check that a macroeconomic shock has been initiated so that we have all required parameters
        if self.severe_outcome_mu_G is None:
            raise Exception('You need to initiate a macroeconomic shock before you can run Monte-Carlo simulations.')

        # We instantiate the dictionary that will store results of the simulation
        results = {
            'n_have_shirked': [],
            'n_have_shirked_or_neg_NPV': [],
            'n_first_best_closures': [],
            'n_capital_requirements_closure': [],
            'n_first_best_balanced_closures_under_shock': [],
            'n_first_best_prudent_closures_under_shock': [],
            'n_capital_requirements_balanced_closures_under_shock': [],
            'n_capital_requirements_prudent_closures_under_shock': [],
            'realized_outcome': [],
            'n_have_shirked_post_shock': [],
            'n_have_shirked_or_neg_NPV_post_shock': [],
            'n_first_best_closures_post_shock': [],
            'n_capital_requirements_closures_post_shock': []
        }

        # We add a progress bar to the for loop, indicating remaining computation time
        for _ in tqdm(range(n_trials)):
            # The following describe what happens for each trial
            # We start by instantiating an economy with the "normal time" parameters
            economy = Economy(
                b=self.b, r=self.r,
                mu_G=self.mu_G, sigma_G=self.sigma_G,
                mu_B=self.mu_B, sigma_B=self.sigma_B,
                lambda_parameter=self.lambda_parameter
            )

            # We run a first simulation with the n_banks argument passed to the method
            economy.run_first_simulation(n_banks=n_banks, fix_random_state=False)

            # We deduce how many banks have shirked or reached a negative NPV, storing the relevant results
            results['n_have_shirked'].append(economy.simulation['has_shirked'].sum())
            results['n_have_shirked_or_neg_NPV'].append(economy.simulation['has_shirked_or_neg_NPV'].sum())

            # We apply the first-best closure of the regulator
            economy.apply_first_best_closure(verbose=0)

            # And we store the number of banks being closed
            results['n_first_best_closures'].append(economy.simulation['first_best_closure'].sum())

            # We apply the second-best or capital requirements closure of the regulator
            economy.apply_capital_requirements(verbose=0)

            # And we store the number of banks being closed
            results['n_capital_requirements_closure'].append(economy.simulation['capital_requirements_closure'].sum())

            # We initiate the macroeconomic shock with the same parameters as the initial Economy instance
            economy.initiate_macro_shock(
                severe_outcome_mu_G=self.severe_outcome_mu_G,
                severe_outcome_sigma_G=self.severe_outcome_sigma_G,
                severe_outcome_mu_B=self.severe_outcome_mu_B,
                severe_outcome_sigma_B=self.severe_outcome_sigma_B,
                light_outcome_mu_G=self.light_outcome_mu_G,
                light_outcome_sigma_G=self.light_outcome_sigma_G,
                light_outcome_mu_B=self.light_outcome_mu_B,
                light_outcome_sigma_B=self.light_outcome_sigma_B,
                verbose=0
            )

            # We apply the balanced first-best closure threshold of the regulator on impact and store relevant results
            df = economy.apply_first_best_closure_under_shock(strategy='balanced', inplace=False, verbose=0)
            df = df[~df['first_best_closure']].copy()
            results['n_first_best_balanced_closures_under_shock'].append(df['first_best_closure_under_shock'].sum())

            # We apply the prudent first-best closure threshold of the regulator on impact and store relevant results
            df = economy.apply_first_best_closure_under_shock(strategy='prudent', inplace=False, verbose=0)
            df = df[~df['first_best_closure']].copy()
            results['n_first_best_prudent_closures_under_shock'].append(df['first_best_closure_under_shock'].sum())

            # We apply the balanced capital requirements closure threshold of the regulator on impact and store results
            df = economy.apply_capital_requirements_under_shock(strategy='balanced', inplace=False, verbose=0)
            df = df[~df['capital_requirements_closure']].copy()
            results['n_capital_requirements_balanced_closures_under_shock'].append(
                df['capital_requirements_closure_under_shock'].sum()
            )

            # We apply the prudent capital requirements closure threshold of the regulator on impact and store results
            df = economy.apply_capital_requirements_under_shock(strategy='prudent', inplace=False, verbose=0)
            df = df[~df['capital_requirements_closure']].copy()
            results['n_capital_requirements_prudent_closures_under_shock'].append(
                df['capital_requirements_closure_under_shock'].sum()
            )

            # We simulate post-shock cash flows of the banks
            economy.simulate_macro_shock(n_periods=200, fix_random_state=False, inplace=True)

            # We store the realized outcome in the results dictionary
            results['realized_outcome'].append(economy.realized_outcome)

            # We deduce how many banks have shirked (after the shock only)
            df = economy.simulation[~economy.simulation['has_shirked']].copy()
            results['n_have_shirked_post_shock'].append(df['has_shirked_post_shock'].sum())

            # We deduce how many banks have shirked or reached a negative NPV (after the shock only)
            df = economy.simulation[~economy.simulation['has_shirked_or_neg_NPV']].copy()
            results['n_have_shirked_or_neg_NPV_post_shock'].append(df['has_shirked_or_neg_NPV_post_shock'].sum())

            # We apply the balanced first-best closure threshold of the regulator on post-shock cash flows
            # And we store the number of previously sound banks (based on first-best threshold) that should be closed
            df = economy.apply_first_best_closure_post_shock(inplace=False, verbose=0)
            df = df[~df['first_best_closure']].copy()
            results['n_first_best_closures_post_shock'].append(df['first_best_closure_post_shock'].sum())

            # We apply the balanced capital requirements closure threshold of the regulator on post-shock cash flows
            # And we store the number of previously sound banks (based on second-best threshold) that should be closed
            df = economy.apply_capital_requirements_post_shock(inplace=False, verbose=0)
            df = df[~df['capital_requirements_closure']].copy()
            results['n_capital_requirements_closures_post_shock'].append(
                df['capital_requirements_closure_post_shock'].sum()
            )

        # We build the DataFrame from results stored in the results dictionary
        df = pd.DataFrame.from_dict(results)

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # monte_carlo_simulation attribute of the Economy instance is updated
            self.monte_carlo_simulation = df.copy()

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('monte_carlo_simulation attribute of the Economy instance was updated (inplace=True passed).')

        else:
            # The attribute is left unchanged and the output is directly returned
            # We print or not the related message depending on the verbose argument
            if verbose:
                print('monte_carlo_simulation attribute was left unchanged (inplace=False was passed).')

            return df.copy()

    def fetch_presaved_monte_carlo_simulation(self, file_id, inplace=True, verbose=1):
        """
        Because computations required by the Monte-Carlo simulation can be heavy, this method allows to fetch the re-
        sults of simulations already run and stored online.

        It requires three arguments:

        - file_id, which indicates what pre-saved simulation results to fetch. Following options are available so far:

            - 0 gives access to a Monte-Carlo simulation run with 250 trials, with 200 banks each. The set of paramaters
            used corresponds to the past calibration (more details can be found in the README file inside the simula-
            tions folder);

            - 1 gives access to a Monte-Carlo simulation run with 250 trials, with 500 banks each. Here again, the set
            of parameters corresponds to the first one used;

            - 2 gives access to a Monte-Carlo simulaton run with 250 trials, with 500 banks each. The set of parameters
            used corresponds to the latest version of the calibration;

            - 3 gives access to a Monte-Carlo simulaton run with 250 trials, with 500 banks each. The set of parameters
            used corresponds to the latest version of the calibration and it includes an additional column, indicating
            the outcome realized during the simulation of the macroeconomic shock for each trial.

        - inplace (True by default), which indicates whether to store the output in the monte_carlo_simulation attribute
        of the Economy instance without returning anything (True) or to return the output instead (False);

        - verbose (1 by default), which determines whether to print or not a message indicating that the attributes have
        been updated, as well as the threshold actually applied.
        """

        # We read the corresponding csv file (path being given by the dictionary defined above)
        df = pd.read_csv(MONTE_CARLO_SIMULATION_PATHS[file_id])

        # We output the result, in two different ways depending on the inplace argument
        if inplace:
            # monte_carlo_simulation attribute of the Economy instance is updated
            self.monte_carlo_simulation = df.copy()

            # We print or not the related message depending on the verbose argument
            if verbose:
                print('monte_carlo_simulation attribute of the Economy instance was updated (inplace=True passed).')

        else:
            # The attribute is left unchanged and the output is directly returned
            # We print or not the related message depending on the verbose argument
            if verbose:
                print('monte_carlo_simulation attribute was left unchanged (inplace=False was passed).')

            return df.copy()

    def plot_monte_carlo_histograms(self):
        """
        This function allows to plot histograms that describe the results of the Monte-Carlo simulation.

        For each of the columns of the monte_carlo_simulation DataFrame, it returns a histogram of the results, with
        a kernel density estimation of the probability density function of the variable when it can be computed.
        """

        # We run a check to verify that a Monte-Carlo simulation has been run in the first place
        if self.monte_carlo_simulation is None:
            raise Exception('You first need to run a Monte-Carlo simulation before plotting its histograms.')

        # We instantiate the figure and the axes for the graphs
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(17, 20))

        # We isolate variables whose distribution we want to plot
        columns = self.monte_carlo_simulation.drop(columns='realized_outcome').columns

        # We iterate over axes and columns of the monte_carlo_simulation DataFrame
        for ax, column_name in zip(axes.flatten(), columns):

            # And we plot the histogram and kernel density estimation for the considered column
            sns.distplot(self.monte_carlo_simulation[column_name], ax=ax)

        # We return the graphs
        plt.show()
