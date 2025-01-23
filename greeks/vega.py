import numpy as np
from pricer.asian import pricer_asian
from pricer.monte_carlo import monte_carlo_simulations
import matplotlib.pyplot as plt
from constants import pricer_mapping
from custom_templates import cyborg_template
import plotly.graph_objects as go

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
    price_S = prices_S['price_call']
    price_S_sigma_h= prices_S_sigma_h['price_call']

    # Compute Vega for call options
    vega = (price_S_sigma_h - price_S) / h

    # Note: we know that vega_call = vega_put, so we note it just 'vega'
    # But in generic functions for greeks, like greek_vs_stock_price, greek_vs_strike_price...
    # We expect as input a dictionary {'greek_call': greek_call, 'greek_put': greek_put}
    return {'vega_call': vega,
            'vega_put': vega}

def vega_vs_stock_price(Z: np.ndarray, 
                         S0_range: np.ndarray, 
                         K: float, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         h: float, 
                         exotic_type: str,
                         n_simulations: int = 100000,
                         **kwargs) -> dict:
    """
    Compute Vega (call and put) as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Vega calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Vegas for calls and puts over the stock price range:
              {'stock_price': S0_range, 'vega_call': vega_call_list, 'vega_put': vega_put_list}.
    """

    vega_list = []

    for S0 in S0_range:
        # Compute Vega for each stock price
        vegas = compute_vega(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        vega_list.append(vegas['vega'])

    return {
        'stock_price': S0_range,
        'vega': vega_list
    }

# def theta_vs_strike_price()

# NOTE: ega is same for call and put, so get rid of all the put part, no _call and _put, and mention why in the info_note
def plot_vega_vs_stock_price(Z: np.ndarray, 
                              S0_range: np.ndarray, 
                              K: float, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              h: float, 
                              exotic_type: str,
                              n_simulations: int = 100000,
                              **kwargs):
    """
    Plot Vega (call and put) as a function of stock price (S0) using Plotly.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    """
    # Compute Vega vs Stock Price
    results = vega_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Vega
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['vega'],
        mode='lines+markers',
        name='Vega'
    ))

    # Add horizontal line at y=0
    fig.add_trace(go.Scatter(
        x=[min(S0_range), max(S0_range)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False  # Do not show in legend
    ))

    # Update layout
    fig.update_layout(
        title="Vega vs Stock Price",
        xaxis_title="Stock Price (S0)",
        yaxis_title="Vega",
        margin=dict(
        l=20,  # Left margin
        r=20,  # Right margin
        t=50,  # Top margin
        b=10   # Bottom margin
        ),
        template=cyborg_template
    )

    # # Show the figure
    # fig.show()

    return(fig)


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


