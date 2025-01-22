import numpy as np
from pricer.asian import pricer_asian
from pricer.monte_carlo import monte_carlo_simulations
import matplotlib.pyplot as plt
from constants import pricer_mapping

# TODO: 
    # compute_vega
    # vega_vs_stock_price
    # vega_vs_ttm
    # vega_vs_vola
    # plot_vega_vs_stock_price
    # plot_3d_vega_vs_diff_implied_vol
    # plot_3d_vega_over_time

def compute_vega(Z: np.ndarray, 
                 S0: float,
                 K: float, 
                 T: float, 
                 r: float, 
                 sigma: float, 
                 h: float,
                 n_simulations: int = 100000) -> float:
    """
    Compute Vega for an Asian option using finite differences.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for finite difference calculation.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.

    Returns:
        dict: Vega for call and put options:
              {'vega_call': vega_call, 'vega_put': vega_put}.
    """
    # Compute option prices at sigma
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    prices_S = pricer_asian(S, K, T, r)

    # Compute option prices at sigma + h
    S_sigma_h = monte_carlo_simulations(Z, S0, T, r, sigma + h, n_simulations)
    prices_S_sigma_h = pricer_asian(S_sigma_h, K, T, r)

    # Extract prices for call options
    price_call_S = prices_S['price_call']
    price_call_S_sigma_h = prices_S_sigma_h['price_call']

    # Compute Vega for call options
    vega_call = (price_call_S_sigma_h - price_call_S) / h

    # Extract prices for put options
    price_put_S = prices_S['price_put']
    price_put_S_sigma_h = prices_S_sigma_h['price_put']

    # Compute Vega for put options
    vega_put = (price_put_S_sigma_h - price_put_S) / h

    return {'vega_call': vega_call,
            'vega_put': vega_put}


def compute_vega(Z: np.ndarray, 
                 S0: float,
                 K: float, 
                 T: float, 
                 r: float, 
                 sigma: float, 
                 h: float,
                 exotic_type: str,
                 n_simulations: int = 100000,
                **kwargs) -> float:
    """
    Compute Vega for an exotic option using the appropriate pricer.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small decrement for finite difference calculation (time decay).
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Vega for call and put options:
              {'vega_call': vega_call, 'vega_put': vega_put}.
    """

    # Fetch the pricer from the mapping
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    # Compute option prices at sigma
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)

    # Compute option prices at sigma + h
    S_sigma_h = monte_carlo_simulations(Z, S0, T, r, sigma + h, n_simulations)

    # Price options using the appropriate pricer
    prices_S = pricer(S, K, T, r, **kwargs)
    prices_S_sigma_h = pricer(S_sigma_h, K, T, r, **kwargs)

    # Extract prices for call options
    price_call_S = prices_S['price_call']
    price_call_S_sigma_h = prices_S_sigma_h['price_call']

    # Compute Vega for call options
    vega_call = (price_call_S_sigma_h - price_call_S) / h

    # Extract prices for put options
    price_put_S = prices_S['price_put']
    price_put_S_sigma_h = prices_S_sigma_h['price_put']

    # Compute Vega for put options
    vega_put = (price_put_S_sigma_h - price_put_S) / h

    return {'vega_call': vega_call,
            'vega_put': vega_put}

def vega_vs_ttm():
    pass

def vega_vs_ttm():
    pass

def vega_vs_vol():
    pass

if __name__ == "__main__":
    # Parameters
    S0 = 100  # Initial stock price
    T = 1  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    K = 100  # Strike price
    h = 0.01  # Small increment for finite differences
    n_simulations = 100000
    exotic_type = "asian" 

    # Generate Z once
    Z = np.random.standard_normal((n_simulations, 252))

    # Compute Vega
    vega = compute_vega(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations)
    print(f"Vega for Asian Call Option: {vega['vega_call']:.6f}")
    print(f"Vega for Asian Put Option: {vega['vega_put']:.6f}")


