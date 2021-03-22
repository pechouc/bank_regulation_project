# Simulations "à la Monte-Carlo"

This folder gathers the output of several Monte-Carlo simulations run with various numbers of trials, numbers of banks or sets of parameters.

As described in the [documentation](https://pechouc.github.io/bank_regulation_project/economy.html#bank_regulation_project.economy.Economy.fetch_presaved_monte_carlo_simulation), these files can be accessed as follows:

```
from bank_regulation_project import Economy
economy = Economy()
economy.fetch_presaved_monte_carlo_simulation(file_id=...)
```

The present document describes in more details the characteristics of each simulation so that you can choose the most relevant one for your study.

## First calibration

### Set of parameters

The first two simulations were run with the following set of parameters:

- "normal time" parameters:

    - `mu_G`: 0.12
    - `sigma_G`: 0.1
    - `mu_B`: -0.03
    - `sigma_B`: 0.2

- "severe outcome" parameters:

    - `mu_G`: 0.09
    - `sigma_G`: 0.1
    - `mu_B`: -0.05
    - `sigma_B`: 0.25

- "light outcome" parameters:

    - `mu_G`: 0.1
    - `sigma_G`: 0.12
    - `mu_B`: -0.04
    - `sigma_B`: 0.22

- other paramaters:

    - `b` (monitoring cost flow): 1.5
    - `r` (interest rate): 0.5
    - `lambda_parameter` (liquidation multiplier): 1.9
    - `d` (share of deposits in the bank's liabilities): 1
    - `severe_outcome_proba`: 0.2


### File references

This calibration is used in the following files:

- `monte_carlo_output_03_10_200_banks.csv` (`file_id=0`), which gathers 250 trials with 200 banks each;

- `monte_carlo_output_03_11_500_banks.csv` (`file_id=1`), which gathers 250 trials with 500 banks each.


## Second calibration

### Update to the set of parameters

The second calibration was essentially introduced to correct for the higher instantaneous variance of the good technology motion in the light outcome than in the severe outcome.

The third simulation was therefore run with the following modifications:

- "severe outcome" parameters:

    - `sigma_G`: 0.12

- "light outcome" parameters:

    - `sigma_G`: 0.11

### File reference

This calibration is used in `monte_carlo_output_03_21_500_banks.csv` (`file_id=2`), which gathers 250 trials with 500 banks each.
