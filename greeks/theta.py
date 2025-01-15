import numpy as np
from asian import pricer_asian
from monte_carlo import monte_carlo_simulations
import matplotlib.pyplot as plt

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
                  n_simulations: int = 100000) -> float:
    """
    Compute Theta for an Asian option using finite differences.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small decrement for finite difference calculation (time decay).
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.

    Returns:
        dict: Theta for call and put options:
              {'theta_call': theta_call, 'theta_put': theta_put}.
    """
    # Compute option prices at T
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    prices_T = pricer_asian(S, K, T, r)

    # Compute option prices at T - h
    S_T_minus_h = monte_carlo_simulations(Z, S0, T - h, r, sigma, n_simulations)
    prices_T_minus_h = pricer_asian(S_T_minus_h, K, T - h, r)

    # Extract prices for call options
    price_call_T = prices_T['price_call']
    price_call_T_minus_h = prices_T_minus_h['price_call']

    # Compute Theta for call options
    theta_call = (price_call_T_minus_h - price_call_T) / h

    # Extract prices for put options
    price_put_T = prices_T['price_put']
    price_put_T_minus_h = prices_T_minus_h['price_put']

    # Compute Theta for put options
    theta_put = (price_put_T_minus_h - price_put_T) / h

    return {'theta_call': theta_call,
            'theta_put': theta_put}

def theta_vs_ttm():
    pass

def theta_vs_ttm():
    pass

def theta_vs_vol():
    pass

if __name__ == "__main__":
    # Parameters
    S0 = 100  # Initial stock price
    T = 1  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    K = 100  # Strike price
    h = 0.01  # Small decrement for time decay
    n_simulations = 100000

    # Generate Z once
    Z = np.random.standard_normal((n_simulations, 252))

    # Compute Theta
    theta = compute_theta(Z, S0, K, T, r, sigma, h, n_simulations)
    print(f"Theta for Asian Call Option: {theta['theta_call']:.6f}")
    print(f"Theta for Asian Put Option: {theta['theta_put']:.6f}")
