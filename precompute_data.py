import numpy as np
import joblib
from constants import N_SIMULATIONS
from pricer_plotter.monte_carlo import monte_carlo_simulations
from greeks.greeks_functions import greek_vs_stock_price
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



# def precompute_results_greek_vs_stock_price():

#     all_exotic_greeks_results = {}

#     N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h = default_input_values()

#     Z = np.random.standard_normal((N_SIMULATIONS, n_steps))

#     S0_range = np.linspace(50, 150, 5)

#     EXOTIC_OPTION_TYPES = ["asian", "lookback", "european"]  #later for barrier as tehy need kwargs

#     GREEKS = ["delta", "gamma", "theta", "vega", "rho"]

#     for exotic in EXOTIC_OPTION_TYPES:
#         print(exotic)

#         # Ensure the key for the exotic option exists as an empty dictionary
#         greeks_results = {}

#         for greek in GREEKS:
#             print(greek)

#             results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek)

#             greeks_results[greek] = results

#         all_exotic_greeks_results[exotic] = greeks_results

#         print('---------------------')

#     return all_exotic_greeks_results

JOBLIB_GREEKS_FILE = "all_exotic_greeks_results.joblib"
# modified output structure to match will the format of stored data in the calbakc 
def precompute_results_greek_vs_stock_price():
    """
    Precomputes Greek vs Stock Price results for all exotic options, including a special case for barrier options.
    Saves the results in a joblib file for fast access.
    """
    all_results_list = []  # Flat list for Dash callback

    # Fetch default input values
    N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h = default_input_values()

    # Generate random values for Z
    Z = np.random.standard_normal((N_SIMULATIONS, n_steps))

    # Define ranges for stock price and strike price
    S0_range = np.linspace(50, 150, 5)
    K_range = np.linspace(50, 150, 5)  # Strike price range

    # Define exotic options and Greeks
    EXOTIC_OPTION_TYPES = ["asian", "lookback", "european", "barrier"]  # Now includes "barrier"
    GREEKS = ["delta", "gamma", "theta", "vega", "rho"]

    # Compute results for each exotic type and Greek
    for exotic in EXOTIC_OPTION_TYPES:
        print(f"Precomputing for {exotic}...")

        for greek in GREEKS:
            print(f"Computing {greek}...")

            if exotic == "barrier":
                # Special case for barrier options (requires additional barrier inputs)
                results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
            else:
                # Standard case for other exotic options
                results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek)

            # Append to flat list
            all_results_list.append(results)

        print('---------------------')

    # Save results in a joblib file for future use
    joblib.dump(all_results_list, JOBLIB_GREEKS_FILE)
    print(f"Precomputed results saved to {JOBLIB_GREEKS_FILE}")

    return all_results_list  # Flat list matching Dash callback output


if __name__ == '__main__':
        


    all_exotic_greeks_results = precompute_results_greek_vs_stock_price()
    print('---------------------')
    joblib.dump(all_exotic_greeks_results, "all_exotic_greeks_results.joblib")
