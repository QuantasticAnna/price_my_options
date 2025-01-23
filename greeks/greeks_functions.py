# Functions common to all greeks, that Im gonna re factor to avoid code duplciation 

# For each greek: compute_greek (specific)
# General: 
#   delta_vs_stock_price, 
#   delta_vs_strike_price, 
#   plot_delta_vs_stock_price,
#   plot_delta_vs_strike_price

import plotly.graph_objects as go
from greeks.greeks_map import greeks_mapping
from plotly.graph_objects import Surface
from custom_templates import cyborg_template
import numpy as np


def greek_vs_stock_price(Z: np.ndarray, 
                         S0_range: np.ndarray, 
                         K: float, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         h: float, 
                         exotic_type: str,
                         greek: str, 
                         n_simulations: int = 100000,
                         **kwargs) -> dict:
    """
    Compute any greek (call and put) as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Greek calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Greeks for calls and puts over the stock price range:
              {'stock_price': S0_range, 'greek_call': greek_call_list, 'greek_put': greek_put_list}.
    """

    # Fetch the greek from the mapping
    compute_greek = greeks_mapping.get(greek)
    if greek is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    greek_call_list = []
    greek_put_list = []

    for S0 in S0_range:
        # Compute Delta for each stock price
        greeks = compute_greek(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        greek_call_list.append(greeks[str(greek) + '_call'])
        greek_put_list.append(greeks[str(greek) + '_put'])

    return {
        'stock_price': S0_range,
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_stock_price(Z: np.ndarray, 
                              S0_range: np.ndarray, 
                              K: float, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              h: float, 
                              exotic_type: str,
                              greek: str, 
                              n_simulations: int = 100000,
                              **kwargs):
    """
    Plot any greek (call and put) as a function of stock price (S0) using Plotly.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    """
    
    # Compute Greek vs Stock Price
    results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, greek, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Greek
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results[str(greek) + '_call'],
        mode='lines+markers',
        name='Call ' + str(greek).capitalize()
    ))

    # Add trace for Put Greek
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results[str(greek) + '_put'],
        mode='lines+markers',
        name='Put ' + str(greek).capitalize()
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
        title=str(greek).capitalize() + " vs Stock Price",
        xaxis_title="Stock Price (S0)",
        yaxis_title=str(greek).capitalize(),
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

    return(fig)


def greek_vs_strike_price(Z: np.ndarray, 
                         S0: float, 
                         K_range: np.ndarray, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         h: float, 
                         exotic_type: str,
                         greek: str, 
                         n_simulations: int = 100000,
                         **kwargs) -> dict:
    """
    Compute any greek (call and put) as a function of STRIKE price (K).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Stock price
        K_range (np.array): Array of strike prices to evaluate.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for greek calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: greeks for calls and puts over the STRIKE price range:
              {'stock_price': S0_range, 'greek_call': greek_call_list, 'greek_put': greek_put_list}.
    """

    # Fetch the greek from the mapping
    compute_greek = greeks_mapping.get(greek)
    if greek is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    greek_call_list = []
    greek_put_list = []

    for K in K_range:
        # Compute Delta for each stock price
        greeks = compute_greek(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        greek_call_list.append(greeks[str(greek) + '_call'])
        greek_put_list.append(greeks[str(greek) + '_put'])

    return {
        'strike_price': K_range,
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_strike_price(Z: np.ndarray, 
                              S0: float, 
                              K_range: np.ndarray, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              h: float, 
                              exotic_type: str,
                              greek: str, 
                              n_simulations: int = 100000,
                              **kwargs):
    """
    Plot any greek (call and put) as a function of strike price (K) using Plotly.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Stock price
        K_range (np.array): Array of strike prices to evaluate.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    """
    
    # Compute Greek vs Stock Price
    results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Greek
    fig.add_trace(go.Scatter(
        x=results['strike_price'],
        y=results[str(greek) + '_call'],
        mode='lines+markers',
        name='Call ' + str(greek).capitalize()
    ))

    # Add trace for Put Greek
    fig.add_trace(go.Scatter(
        x=results['strike_price'],
        y=results[str(greek) + '_put'],
        mode='lines+markers',
        name='Put ' + str(greek).capitalize()
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
        title=str(greek).capitalize() + " vs Strike Price",
        xaxis_title="Strike Price (S0)",
        yaxis_title=str(greek).capitalize(),
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

    return(fig)

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

    # results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, greek)

    # plot_stock = plot_greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, greek)
    # plot_stock.show()

    # results_strike = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek)
    plot_strike = plot_greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek)
    plot_strike.show()
    