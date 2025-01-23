import numpy as np
from greeks.delta import compute_delta 

# Note: we spend a lot of time computing delta for multiple stocks prices, 
# and probably here we compute it again for the same prices, 
# think abolult a way to do all the simulation for different stocks prices once, and then compute delta, gamma and delta vs stock price and gamma vs stock price

# TODO: 
    # compute_gamma
    # gamma_vs_stock_price
    # gamma_vs_ttm
    # gamma_vs_vola
    # plot_gamma_vs_stock_price
    # plot_3d_gamma_vs_diff_implied_vol
    # plot_3d_gamma_over_time

    # Implement note above

def compute_gamma(Z: np.ndarray, 
                  S0: float, 
                  K: float, 
                  T: float, 
                  r: float, 
                  sigma: float, 
                  h: float, 
                  exotic_type = str,
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
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.

    Returns:
        dict: Gamma for call and put options:
              {'gamma_call': gamma_call, 'gamma_put': gamma_put}.
    """
    # Gamma_call = gamma_put, so we dont need to make two separate cases 

    # Compute Delta at S0
    delta_S0 = compute_delta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations)['delta_call'] 

    # Compute Delta at S0 + h
    delta_S0_h = compute_delta(Z, S0 + h, K, T, r, sigma, h, exotic_type, n_simulations)['delta_call']

    # Compute Gamma via finite difference
    gamma = (delta_S0_h - delta_S0) / h

    return {'gamma_call': gamma,
            'gamma_put': gamma}



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
    compute_gamma(Z, S0_range, K, T, r, sigma, h, n_simulations)

    # gamma plot has the correct expected bell shape, but is very ugly, probably because not enough monte carlo simulation
    # as gamma is the second derivative, but if we put more simulations, then eveything else will be slowed down 


#! Gonna be a bit annoying to plot gamma for differnet TTm and different vol, as i need to make more simulation to have a nice gamma (for one TTm and one vol), and it really increases computation time 