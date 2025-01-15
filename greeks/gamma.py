import numpy as np
from delta import compute_delta 

# Note: we spend a lot of time computing delta for multiple stocks prices, 
# and probably here we compute it again for the same prices, 
# think abolult a way to do all the simulation for different stocks prices once, and then compute delta, gamma and delta vs stock price and gamma vs stock price


def compute_gamma(Z: np.ndarray, 
                  S0: float, 
                  K: float, 
                  T: float, 
                  r: float, 
                  sigma: float, 
                  h: float, 
                  n_simulations: int = 100000) -> float:
    """
    Compute Gamma for an Asian option using finite differences.

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
        dict: Gamma for call and put options:
              {'gamma_call': gamma_call, 'gamma_put': gamma_put}.
    """
    # Gamma_call = gamma_put, so we dont need to make two separate cases 

    # Compute Delta at S0
    delta_S0 = compute_delta(Z, S0, K, T, r, sigma, h, n_simulations)['delta_call'] 

    # Compute Delta at S0 + h
    delta_S0_h = compute_delta(Z, S0 + h, K, T, r, sigma, h, n_simulations)['delta_call']

    # Compute Gamma via finite difference
    gamma = (delta_S0_h - delta_S0) / h

    return gamma


def gamma_vs_stock_price(Z: np.ndarray, 
                         S0_range: np.ndarray, 
                         K: float, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         h: float, 
                         n_simulations: int = 100000) -> dict:
    """
    Compute Gamma as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for finite difference calculation.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.

    Returns:
        dict: Gamma values over the stock price range:
              {'stock_price': S0_range, 'gamma': gamma_list}.
    """
    gamma_list = []

    for S0 in S0_range:
        gamma = compute_gamma(Z, S0, K, T, r, sigma, h, n_simulations)
        gamma_list.append(gamma)

    return {'stock_price': S0_range, 'gamma': gamma_list}

import matplotlib.pyplot as plt

def plot_gamma_vs_stock_price(Z: np.ndarray, 
                              S0_range: np.ndarray, 
                              K: float, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              h: float, 
                              n_simulations: int = 100000):
    """
    Plot Gamma as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for finite difference calculation.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
    """
    # Compute Gamma vs Stock Price
    results = gamma_vs_stock_price(Z, S0_range, K, T, r, sigma, h, n_simulations)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results['stock_price'], results['gamma'], label="Gamma", marker='o')
    plt.xlabel("Stock Price (S0)")
    plt.ylabel("Gamma")
    plt.title("Gamma vs Stock Price")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Parameters
    T = 1  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    K = 100  # Strike price
    h = 0.01  # Small increment for finite differences
    n_simulations = 100000

    # Define stock price range
    S0_range = np.linspace(50, 150, 20)  # Stock prices from 50 to 150

    # Generate Z once
    Z = np.random.standard_normal((n_simulations, 252))

    # Plot Gamma vs Stock Price
    plot_gamma_vs_stock_price(Z, S0_range, K, T, r, sigma, h, n_simulations)

    # gamma plot has the correct expected bell shape, but is very ugly, probably because not enough monte carlo simulation
    # as gamma is the second derivative, but if we put more simulations, then eveything else will be slowed down 

