import numpy as np
from pricer.monte_carlo import monte_carlo_simulations
from constants import pricer_mapping

def compute_rho(Z: np.ndarray, 
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
    Compute Rho for an exotic option using the appropriate pricer.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for finite difference calculation (interest rate sensitivity).
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "B_call" and "B_put" for barrier options).
    
    Returns:
        dict: Rho for call and put options:
              {'rho_call': rho_call, 'rho_put': rho_put}.
    """
    # Fetch the pricer from the mapping
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    # Simulate prices
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_r_plus_h = monte_carlo_simulations(Z, S0, T, r + h, sigma, n_simulations)

    # Extract additional parameters for barrier options, if applicable
    if exotic_type == 'barrier':
        B_call = kwargs.get("B_call")
        B_put = kwargs.get("B_put")
        if B_call is None or B_put is None:
            raise ValueError("Barrier parameters 'B_call' and 'B_put' are required for barrier options.")
        prices_r = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
        prices_r_plus_h = pricer(S_r_plus_h, K, T, r, B_call=B_call, B_put=B_put)
    else:
        # Price options using the standard pricer
        prices_r = pricer(S, K, T, r, **kwargs)
        prices_r_plus_h = pricer(S_r_plus_h, K, T, r, **kwargs)

    # Extract prices for call options
    price_call_r = prices_r['price_call']
    price_call_r_plus_h = prices_r_plus_h['price_call']

    # Compute Rho for call options
    rho_call = (price_call_r_plus_h - price_call_r) / h

    # Extract prices for put options
    price_put_r = prices_r['price_put']
    price_put_r_plus_h = prices_r_plus_h['price_put']

    # Compute Rho for put options
    rho_put = (price_put_r_plus_h - price_put_r) / h

    return {'rho_call': rho_call,
            'rho_put': rho_put}



if __name__ == "__main__":
    # Parameters
    S0 = 100  # Initial stock price
    T = 1  # Time to maturity (1 year)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    K = 100  # Strike price
    h = 0.0001  # Small increment for finite differences
    n_simulations = 100000

    # Generate Z once
    Z = np.random.standard_normal((n_simulations, 252))

    # Compute Rho
    rho = compute_rho(Z, S0, K, T, r, sigma, h, 'asian', n_simulations)
    print(f"Rho for Asian Call Option: {rho['rho_call']:.6f}")
    print(f"Rho for Asian Put Option: {rho['rho_put']:.6f}")
