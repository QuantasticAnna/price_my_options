import numpy as np
from pricer.asian import pricer_asian
from pricer.monte_carlo import monte_carlo_simulations
import matplotlib.pyplot as plt
from constants import pricer_mapping
from custom_templates import cyborg_template
import plotly.graph_objects as go


# TODO: 
    # compute_rho
    # rho_vs_stock_price
    # rho_vs_ttm
    # rho_vs_vola
    # plot_rho_vs_stock_price
    # plot_3d_rho_vs_diff_implied_vol
    # plot_3d_rho_over_time


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
        h (float): Small decrement for finite difference calculation (time decay).
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    
    Returns:
        dict: Rho for call and put options:
              {'rho_call': rho_call, 'rho_put': rho_put}.
    """
    # Fetch the pricer from the mapping
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    # Compute option prices at r
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    prices_r = pricer(S, K, T, r, **kwargs)

    # Compute option prices at r + h
    S_r_plus_h = monte_carlo_simulations(Z, S0, T, r + h, sigma, n_simulations)
    prices_r_plus_h = pricer(S_r_plus_h, K, T, r + h, **kwargs)

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


def rho_vs_stock_price(Z: np.ndarray, 
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
    Compute Rho (call and put) as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility
        h (float): Small increment for Rho calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Rhos for calls and puts over the stock price range:
              {'stock_price': S0_range, 'rho_call': rho_call_list, 'rho_put': rho_put_list}.
    """
    rho_call_list = []
    rho_put_list = []

    for S0 in S0_range:
        # Compute Rho for each stock price
        rhos = compute_rho(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        rho_call_list.append(rhos['rho_call'])
        rho_put_list.append(rhos['rho_put'])

    return {
        'stock_price': S0_range,
        'rho_call': rho_call_list,
        'rho_put': rho_put_list
    }


def plot_rho_vs_stock_price(Z: np.ndarray, 
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
    Plot Rho (call and put) as a function of stock price (S0) using Plotly.

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
    # Compute Rho vs Stock Price
    results = rho_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Rho
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['rho_call'],
        mode='lines+markers',
        name='Call Rho'
    ))

    # Add trace for Put Rho
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['rho_put'],
        mode='lines+markers',
        name='Put Rho'
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
        title="Rho vs Stock Price",
        xaxis_title="Stock Price (S0)",
        yaxis_title="Rho",
        legend=dict(
            title="Option Type",
            x=0.5,  # Center horizontally
            y=-0.3,  # Place below the graph
            xanchor="center",  # Anchor legend at its center horizontally
            yanchor="top",  # Anchor legend to the top of its box
            orientation="h"  # Horizontal legend layout
        ),
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

def rho_vs_ttm():
    pass

def rho_vs_ttm():
    pass

def rho_vs_vol():
    pass

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
    rho = compute_rho(Z, S0, K, T, r, sigma, h, n_simulations)
    print(f"Rho for Asian Call Option: {rho['rho_call']:.6f}")
    print(f"Rho for Asian Put Option: {rho['rho_put']:.6f}")
