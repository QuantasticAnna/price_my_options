import numpy as np
from pricer.monte_carlo import monte_carlo_simulations
from constants import pricer_mapping


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
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")

    # Simulate prices
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_h = monte_carlo_simulations(Z, S0 + h, T, r, sigma, n_simulations)

    # Price options using the appropriate pricer
    prices_S = pricer(S, K, T, r, **kwargs)
    prices_S_h = pricer(S_h, K, T, r, **kwargs)

    # Compute Delta for call and put
    delta_call = (prices_S_h['price_call'] - prices_S['price_call']) / h
    delta_put = (prices_S_h['price_put'] - prices_S['price_put']) / h

    return {'delta_call': delta_call, 'delta_put': delta_put}

