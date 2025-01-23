from greeks.greeks_functions import greek_vs_strike_price
import numpy as np


def delta_vs_strike_price_for_multiple_volatility(Z: np.ndarray, 
                                                 S0: float, 
                                                 volatilities: np.ndarray, 
                                                 K_range: np.ndarray, 
                                                 T: float, 
                                                 r: float, 
                                                 h: float, 
                                                 exotic_type: str, 
                                                 n_simulations: int = 100000,
                                                 **kwargs) -> dict:
    """
    Compute a matrix of Delta values for multiple volatilities and stock prices.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        volatilities (np.ndarray): Array of volatility values to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Dictionary containing:
              - 'stock_price': S0_range
              - 'volatility': volatilities
              - 'delta_matrix_call': 2D array of Call Deltas
              - 'delta_matrix_put': 2D array of Put Deltas
    """

    dict_delta_call = {}
    dict_delta_put = {}

    greek = 'delta'

    # Iterate over volatilities
    for sigma in volatilities:
        # Compute Delta for each volatility
        # results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek, n_simulations, **kwargs)
        results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek)

        dict_delta_call[sigma] = results['delta_call']
        dict_delta_put[sigma] = results['delta_put']

    return dict_delta_call, dict_delta_put




def delta_vs_strike_price_for_multiple_ttm(Z: np.ndarray, 
                                                 S0: float, 
                                                 sigma: float, 
                                                 K_range: np.ndarray, 
                                                 TTM_array: np.ndarray, 
                                                 r: float, 
                                                 h: float, 
                                                 exotic_type: str, 
                                                 n_simulations: int = 100000,
                                                 **kwargs) -> dict:
    """
    Compute a matrix of Delta values for multiple volatilities and stock prices.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        sigma (float): Volatility.
        K (float): Strike price.
        TTM_array (np.ndarray): Array of different times to maturiy
        r (float): Risk-free rate.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Dictionary containing:
              - 'stock_price': S0_range
              - 'TTM': times to maturity
              - 'delta_matrix_call': 2D array of Call Deltas
              - 'delta_matrix_put': 2D array of Put Deltas
    """


    dict_delta_call = {}
    dict_delta_put = {}

    # Iterate over volatilities
    for TTM in TTM_array:
        # Compute Delta for each volatility
        results = greek_vs_strike_price(Z, S0, K_range, TTM, r, sigma, h, exotic_type, 'delta', n_simulations, **kwargs)

        dict_delta_call[TTM] = results['delta_call']
        dict_delta_put[TTM] = results['delta_put']

    return dict_delta_call, dict_delta_put




if __name__ == "__main__":
    # Define parameters
    S0_range = np.linspace(50, 150, 20)  # Stock price range (as ndarray)
    S0 = 100
    volatilities = np.array([0.1, 0.2, 0.3])  # Volatility range (as ndarray)
    sigma = 0.1
    Z = np.random.standard_normal((100000, 252))  # Precomputed random normals
    K = 100  # Strike price
    K_range = np.linspace(50, 150, 20)  # Strike price range (as ndarray)
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    h = 0.01  # Small increment for Delta calculation
    exotic_type = "asian"  # Exotic option type
    greek = 'delta'

    # plot_delta_vs_strike_price(Z, 
    #                           S0, 
    #                           K_range, 
    #                           T, 
    #                           r, 
    #                           sigma, 
    #                           h, 
    #                           exotic_type)
    
    # # Compute Delta matrix for multiple volatilities
    dict_to_return_call, dict_to_return_put = delta_vs_strike_price_for_multiple_volatility(
        Z=Z,
        S0=S0,
        volatilities=volatilities,
        K_range=K_range,
        T=T,
        r=r,
        h=h,
        exotic_type=exotic_type,
        greek = greek,
        n_simulations=100000
    )

    dict_to_return_call, dict_to_return_put = delta_vs_strike_price_for_multiple_ttm(
        Z=Z,
        S0=S0,
        volatilities=volatilities,
        K_range=K_range,
        T=T,
        r=r,
        h=h,
        exotic_type=exotic_type,
        greek = greek,
        n_simulations=100000
    )