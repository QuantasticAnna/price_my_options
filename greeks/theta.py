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

    # Price options using the appropriate pricer
    prices_S = pricer(S, K, T, r, **kwargs)
    prices_S_h = pricer(S_T_minus_h, K, T - h, r, **kwargs)

    # Compute Theta for call and put
    theta_call = (prices_S_h['price_call'] - prices_S['price_call']) / h
    theta_put = (prices_S_h['price_put'] - prices_S['price_put']) / h

    return {'theta_call': theta_call, 'theta_put': theta_put}



def theta_vs_stock_price(Z: np.ndarray, 
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
    Compute Theta (call and put) as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.Theta
        h (float): Small increment for Theta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Thetas for calls and puts over the stock price range:
              {'stock_price': S0_range, 'theta_call': theta_call_list, 'theta_put': theta_put_list}.
    """
    theta_call_list = []
    theta_put_list = []

    for S0 in S0_range:
        # Compute Theta for each stock price
        thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        theta_call_list.append(thetas['theta_call'])
        theta_put_list.append(thetas['theta_put'])

    return {
        'stock_price': S0_range,
        'theta_call': theta_call_list,
        'theta_put': theta_put_list
    }

def theta_vs_strike_price(Z: np.ndarray, 
                         S0: float, 
                         K_range: np.ndarray, 
                         T: float, 
                         r: float, 
                         sigma: float, 
                         h: float, 
                         exotic_type: str,
                         n_simulations: int = 100000,
                         **kwargs) -> dict:
    """
    Compute Theta (call and put) as a function of STRIKE price for a fixed stock price using Plotly.

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0_range (np.ndarray): Array of stock prices to evaluate.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Theta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Thetas for calls and puts over the stock price range:
              {'stock_price': S0_range, 'theta_call': theta_call_list, 'theta_put': theta_put_list}.
    """
    theta_call_list = []
    theta_put_list = []

    for K in K_range:
        # Compute Theta for each stock price
        thetas = compute_theta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        theta_call_list.append(thetas['theta_call'])
        theta_put_list.append(thetas['theta_put'])

    return {
        'strike_price': K_range,
        'theta_call': theta_call_list,
        'theta_put': theta_put_list
    }



def plot_theta_vs_stock_price(Z: np.ndarray, 
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
    Plot Theta (call and put) as a function of stock price (S0) using Plotly.

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
    # Compute Theta vs Stock Price
    results = theta_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Theta
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['theta_call'],
        mode='lines+markers',
        name='Call Theta'
    ))

    # Add trace for Put Theta
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['theta_put'],
        mode='lines+markers',
        name='Put Theta'
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
        title="Theta vs Stock Price",
        xaxis_title="Stock Price (S0)",
        yaxis_title="Theta",
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


def theta_vs_ttm():
    pass

def theta_vs_vol():
    pass

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

    results = theta_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, n_simulations)
    print(f"Theta for Asian Call Option: {theta['theta_call']:.6f}")
    print(f"Theta for Asian Put Option: {theta['theta_put']:.6f}")
