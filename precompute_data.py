import numpy as np
import joblib
from constants import N_SIMULATIONS
from pricer_plotter.monte_carlo import monte_carlo_simulations

# Temporary: in the first version of the app, Z can not be recompute, its always the same, to facilitate development
# because when we compute Z from the UI, it takes a lot of time 

# Precompute Z and store it
def generate_Z(n_simulations=N_SIMULATIONS, n_steps=252, filename="Z_precomputed.joblib"):
    Z = np.random.standard_normal((n_simulations, n_steps))
    joblib.dump(Z, filename)
    print(f"Precomputed Z saved to {filename}")


def default_input_values():
    N_SIMULATIONS = 100000
    n_steps = 252
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    B_call = 90
    B_put = 110
    h = 0.01
    # Also ranges for S0, K, TTM?

    return N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h #ranges? 


def precompute_heavy_data(filename="data_precomputed.joblib"):

    N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h = default_input_values()

    print('precomputing Z')
    Z = np.random.standard_normal((N_SIMULATIONS, n_steps))

    print('precomputing S')
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)

    precomputed_data = {'Z' : Z,
                        'S' : S}

    joblib.dump(precomputed_data, filename)

    print(f"Precomputed data saved to {filename}")



if __name__ == '__main__':
        
    # Run this script once to generate and save Z
    # generate_Z()
    precompute_heavy_data(filename="data_precomputed.joblib")