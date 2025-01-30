import numpy as np
from pricer.monte_carlo import monte_carlo_simulations
from constants import PRICER_MAPPING


def compute_delta(Z: np.ndarray, 
                  S0: float,
                  K: float, 
                  T: float, 
                  r: float, 
                  sigma: float, 
                  h: float,
                  exotic_type: str,
                  n_simulations: int = 100000,
                  **kwargs) -> dict:
    """
    Compute Delta for an exotic option using the appropriate pricer.
    
    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    
    Returns:
        dict: Delta for call and put options.
    """
    # Fetch the pricer from the mapping
    pricer = PRICER_MAPPING.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")

    # Simulate prices
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_h = monte_carlo_simulations(Z, S0 + h, T, r, sigma, n_simulations)

    # Add barrier parameters if the pricer is for barrier options
    if exotic_type == 'barrier':
        B_call = kwargs.get("B_call", None)
        B_put = kwargs.get("B_put", None)
        if B_call is None or B_put is None:
            raise ValueError("Barrier parameters B_call and B_put must be provided for barrier options.")
        prices_S = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
        prices_S_h = pricer(S_h, K, T, r, B_call=B_call, B_put=B_put)
    else:
        # Price options using the standard pricer
        prices_S = pricer(S, K, T, r, **kwargs)
        prices_S_h = pricer(S_h, K, T, r, **kwargs)

    # Compute Delta for call and put
    delta_call = (prices_S_h['price_call'] - prices_S['price_call']) / h
    delta_put = (prices_S_h['price_put'] - prices_S['price_put']) / h

    return {'delta_call': delta_call, 'delta_put': delta_put}


if __name__ == '__main__':

    n_simulations = 100000
    n_steps = 252

    Z = np.random.standard_normal((n_simulations, n_steps))

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    h = 0.01
    exotic_type = 'asian'

    results = compute_delta(Z, S0, K, T, r, sigma, h, exotic_type)

    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_h = monte_carlo_simulations(Z, S0 + h, T, r, sigma, n_simulations)

    print(results)