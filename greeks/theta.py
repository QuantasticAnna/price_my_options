import numpy as np
from pricer.asian import pricer_asian
from pricer.monte_carlo import monte_carlo_simulations
import matplotlib.pyplot as plt
from constants import pricer_mapping
from custom_templates import cyborg_template
import plotly.graph_objects as go

# TODO: 
    # compute_theta
    # theta_vs_stock_price
    # theta_vs_ttm
    # theta_vs_vola
    # plot_theta_vs_stock_price
    # plot_3d_theta_vs_diff_implied_vol
    # plot_3d_theta_over_time


def compute_theta(Z: np.ndarray, 
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
    Compute Theta for an exotic option using the appropriate pricer.
    
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
        dict: Theta for call and put options.
    """
    # Fetch the pricer from the mapping
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")

    # Simulate prices
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_T_minus_h = monte_carlo_simulations(Z, S0, T - h, r, sigma, n_simulations)

    # Extract additional parameters for barrier options, if applicable
    if exotic_type == 'barrier':
        B_call = kwargs.get("B_call")
        B_put = kwargs.get("B_put")
        if B_call is None or B_put is None:
            raise ValueError("Barrier parameters 'B_call' and 'B_put' are required for barrier options.")
        prices_S = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
        prices_S_h = pricer(S_T_minus_h, K, T - h, r, B_call=B_call, B_put=B_put)
    else:
        # Price options using the appropriate pricer
        prices_S = pricer(S, K, T, r, **kwargs)
        prices_S_h = pricer(S_T_minus_h, K, T - h, r, **kwargs)

    # Compute Theta for call and put
    theta_call = (prices_S_h['price_call'] - prices_S['price_call']) / h
    theta_put = (prices_S_h['price_put'] - prices_S['price_put']) / h

    return {'theta_call': theta_call, 'theta_put': theta_put}

if __name__ == "__main__":
    # Parameters
    S0 = 100  # Initial stock price
    T = 1  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.05  # Volatility
    K = 100  # Strike price
    h = 0.01  # Small decrement for time decay
    n_simulations = 100000
    exotic_type = "asian" 
    K_range = np.linspace(50, 150, 10)

    # Generate Z once
    Z = np.random.standard_normal((n_simulations, 252))

    # Compute Theta
    theta = compute_theta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations)

    theta_barrier = compute_theta(Z, S0=100, K=100, T=1, r=0.05, sigma=0.2, h=0.01, 
                      exotic_type="barrier", 
                      B_call=90, B_put=110)
    

    print(f"Theta for Asian Call Option: {theta['theta_call']:.6f}")
    print(f"Theta for Asian Put Option: {theta['theta_put']:.6f}")
