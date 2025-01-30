import numpy as np
from pricer_plotter.monte_carlo import monte_carlo_simulations
from constants import PRICER_MAPPING
import plotly.graph_objects as go

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
    pricer = PRICER_MAPPING.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    # Compute option prices at sigma
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)

    # Compute option prices at sigma + h
    S_sigma_h = monte_carlo_simulations(Z, S0, T, r, sigma + h, n_simulations)


    # Add barrier parameters if the pricer is for barrier options
    if exotic_type == 'barrier':
        B_call = kwargs.get("B_call", None)
        B_put = kwargs.get("B_put", None)
        if B_call is None or B_put is None:
            raise ValueError("Barrier parameters B_call and B_put must be provided for barrier options.")
        prices_S = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
        prices_S_sigma_h = pricer(S_sigma_h, K, T, r, B_call=B_call, B_put=B_put)
    else:
        prices_S = pricer(S, K, T, r, **kwargs)
        prices_S_sigma_h = pricer(S_sigma_h, K, T, r, **kwargs)

    # Extract prices for call options
    price_S = prices_S['price_call']
    price_S_sigma_h= prices_S_sigma_h['price_call']

    # Compute Vega for call options
    vega = (price_S_sigma_h - price_S) / h

    # Note: we know that vega_call = vega_put, so we note it just 'vega'
    # But in generic functions for greeks, like greek_vs_stock_price, greek_vs_strike_price...
    # We expect as input a dictionary {'greek_call': greek_call, 'greek_put': greek_put}
    return {'vega_call': vega,
            'vega_put': vega}


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

    vega_barrier = compute_vega(Z, S0=100, K=100, T=1, r=0.05, sigma=0.2, h=0.01, 
                      exotic_type="barrier", 
                      B_call=90, B_put=110)
    print(f"Vega for Asian Call Option: {vega['vega_call']:.6f}")
    print(f"Vega for Asian Put Option: {vega['vega_put']:.6f}")


