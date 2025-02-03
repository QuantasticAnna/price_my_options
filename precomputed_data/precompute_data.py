import numpy as np
import joblib
from pricer_plotter.monte_carlo import monte_carlo_simulations
from greeks.greeks_functions import greek_vs_stock_price, greek_vs_strike_price, greek_vs_ttm
from constants import PRICER_MAPPING
from greeks.delta import compute_delta
from greeks.gamma import compute_gamma
from greeks.vega import compute_vega
from greeks.theta import compute_theta
from greeks.rho import compute_rho

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


def precompute_heavy_data(filename="precomputed_data/data_precomputed.joblib"):

    N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h = default_input_values()

    print('precomputing Z')
    Z = np.random.standard_normal((N_SIMULATIONS, n_steps))

    print('precomputing S')
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)

    precomputed_data = {'Z' : Z,
                        'S' : S}

    joblib.dump(precomputed_data, filename)

    print(f"Precomputed data saved to {filename}")



JOBLIB_GREEKS_VS_STOCK_PRICE_FILE = "precomputed_data/precomputed_greeks_vs_stock_price_results.joblib"
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
    S0_range = np.linspace(50, 150, 30)

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
                results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS, B_call=B_call, B_put=B_put)
            else:
                # Standard case for other exotic options
                results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS)
            # Append to flat list
            all_results_list.append(results)

    # Save results in a joblib file for future use
    joblib.dump(all_results_list, JOBLIB_GREEKS_VS_STOCK_PRICE_FILE)
    print(f"Precomputed results saved to {JOBLIB_GREEKS_VS_STOCK_PRICE_FILE}")

    return all_results_list  # Flat list matching Dash callback output


JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE = "precomputed_data/precomputed_greeks_vs_strike_price_results.joblib"
def precompute_results_greek_vs_strike_price():
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
    K_range = np.linspace(50, 150, 30) 

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
                results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS, B_call=B_call, B_put=B_put)
            else:
                # Standard case for other exotic options
                results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS)
            # Append to flat list
            all_results_list.append(results)

    # Save results in a joblib file for future use
    joblib.dump(all_results_list, JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE)
    print(f"Precomputed results saved to {JOBLIB_GREEKS_VS_STRIKE_PRICE_FILE}")

    return all_results_list  # Flat list matching Dash callback output


JOBLIB_GREEKS_VS_TTM_FILE = "precomputed_data/precomputed_greeks_vs_ttm_results.joblib"
def precompute_results_greek_vs_ttm():
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
    TTM_RANGE = np.linspace(0.1, 1, 10)

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
                results = greek_vs_ttm(Z, S0, K, TTM_RANGE, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS, B_call=B_call, B_put=B_put)
            else:
                # Standard case for other exotic options
                results = greek_vs_ttm(Z, S0, K, TTM_RANGE, r, sigma, h, exotic, greek, n_simulations = N_SIMULATIONS)
            # Append to flat list
            all_results_list.append(results)

    # Save results in a joblib file for future use
    joblib.dump(all_results_list, JOBLIB_GREEKS_VS_TTM_FILE)
    print(f"Precomputed results saved to {JOBLIB_GREEKS_VS_TTM_FILE}")

    return all_results_list  # Flat list matching Dash callback output



JOBLIB_OPTIONS_PRICES_AND_GREEKS = "precomputed_data/precomputed_option_prices_and_greeks.joblib"
def precompute_option_prices_and_greeks():

    N_SIMULATIONS, n_steps, S0, K, T, r, sigma, B_call, B_put, h = default_input_values()
    EXOTIC_OPTION_TYPES = ["asian", "lookback", "european", "barrier"] 

    # Generate random values for Z
    Z = np.random.standard_normal((N_SIMULATIONS, n_steps))

    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)

    results_greeks_prices = {}

    for exotic in EXOTIC_OPTION_TYPES:
        pricer = PRICER_MAPPING.get(exotic)
        # Compute Greeks and Prices
        if exotic == "barrier":
            # Compute Greeks for barrier options
            deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
            gammas = compute_gamma(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
            thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
            vegas = compute_vega(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
            rhos = compute_rho(Z, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)

            # Compute Option Prices
            prices = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
        else:
            # Compute Greeks for other options
            deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic)
            gammas = compute_gamma(Z, S0, K, T, r, sigma, h, exotic)
            thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic)
            vegas = compute_vega(Z, S0, K, T, r, sigma, h, exotic)
            rhos = compute_rho(Z, S0, K, T, r, sigma, h, exotic)

            # Compute Option Prices
            prices = pricer(S, K, T, r)

        results_greeks_prices[exotic] = {'deltas': deltas,
                                        'gammas': gammas,
                                        'thetas': thetas,
                                        'vegas': vegas,
                                        'rhos': rhos,
                                        'prices': prices}

    # Save results in a joblib file for future use
    joblib.dump(results_greeks_prices, JOBLIB_OPTIONS_PRICES_AND_GREEKS)
    print(f"Precomputed results prices and greeks saved to {JOBLIB_OPTIONS_PRICES_AND_GREEKS}")

    return results_greeks_prices  # Flat list matching Dash callback output

if __name__ == '__main__':

    #all_exotic_greeks_results_stock_price = precompute_results_greek_vs_stock_price()
    # all_exotic_greeks_results_strike_price = precompute_results_greek_vs_strike_price()
    # all_exotic_greeks_results_ttm = precompute_results_greek_vs_ttm()

    # precompute_heavy_data()

    results_greeks_prices = precompute_option_prices_and_greeks()

    print('---------------------')

