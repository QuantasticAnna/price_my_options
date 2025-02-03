"""
This script defines various functions for computing and visualizing option Greeks 
(Delta, Gamma, Vega, Theta, Rho) as a function of key option parameters such as stock price, 
strike price, time-to-maturity (TTM), and volatility. It is designed to work with both 
standard and exotic options (e.g., Asian options) using Monte Carlo simulations.

Key Features:
1. **Greek Computations**:
    - `greek_vs_stock_price`: Computes Greeks as a function of the stock price (S0).
    - `greek_vs_strike_price`: Computes Greeks as a function of the strike price (K).
    - `greek_vs_ttm`: Computes Greeks as a function of time to maturity (T).
    - `greek_vs_volatility`: Computes Greeks as a function of volatility (σ).

2. **Plotting**:
    - `plot_greek_vs_stock_price`: Visualizes Greeks against stock price (S0).
    - `plot_greek_vs_strike_price`: Visualizes Greeks against strike price (K).
    - `plot_greek_vs_ttm`: Visualizes Greeks against time to maturity (T).
    - `plot_greek_vs_volatility`: Visualizes Greeks against volatility (σ).

3. **Monte Carlo Integration**:
    - Uses precomputed random normals (`Z`) to simulate option prices and compute Greeks.

4. **Customizable Options**:
    - Supports different exotic option types (e.g., Asian options).
    - Includes flexibility for additional parameters like barrier levels.

Dependencies:
- `plotly`: For creating interactive plots.
- `numpy`: For numerical computations.
- `greeks_mapping`: Maps Greek names to their respective computation functions.
- `cyborg_template`: Custom Plotly template for consistent plot styling.

Usage:
1. Precompute random normals (`Z`) for Monte Carlo simulations.
2. Call the desired Greek computation function (e.g., `greek_vs_stock_price`).
3. Use the corresponding plot function (e.g., `plot_greek_vs_stock_price`) to visualize results.

"""
import plotly.graph_objects as go
from greeks.greeks_map import greeks_mapping
from custom_templates import cyborg_template
import numpy as np
import datetime 


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
              {'stock_price': S0_range,         
                'K': K,
                'r': r,
                'T': T,
                'sigma': sigma,
                'greek': greek, 
                'greek_call': greek_call_list, 
                'greek_put': greek_put_list}.
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
        'K': K,
        'r': r,
        'T': T,
        'sigma': sigma,
        'greek': greek, 
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_stock_price(results,
                              ):
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

    S0_range = results['stock_price']
    K = results['K']
    T = results['T']
    r = results['r']
    sigma = results['sigma']
    greek = results['greek']

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

    # Display input variables as annotations
    input_text = f"K={K:.2f}, T={T:.2f}, r={r:.2%}, σ={sigma:.2f}"
    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=input_text,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

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
              {'strike_price': K_range,
                'S0': S0,
                'r': r,
                'T': T,
                'sigma': sigma,
                'greek': greek, 
                'greek_call': greek_call_list, 
                'greek_put': greek_put_list}.
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
        'S0': S0,
        'r': r,
        'T': T,
        'sigma': sigma,
        'greek': greek, 
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_strike_price(results):
    """
    Plot any greek (call and put) as a function of strike price (K) using Plotly.
    """

    # unpack results in a better way 
    K_range = results['strike_price']
    S0 = results['S0']
    T = results['T']
    r = results['r']
    sigma = results['sigma']
    greek = results['greek']


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
        x=[min(K_range), max(K_range)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False  # Do not show in legend
    ))

    # Display input variables as annotations
    input_text = f"S0={S0:.2f}, T={T:.2f}, r={r:.2%}, σ={sigma:.2f}"
    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=input_text,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

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



def greek_vs_ttm(Z: np.ndarray, 
                S0: float, 
                K: float, 
                T_range: np.ndarray, 
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
        S0 (float): Stock price.
        K (float): Strike price.
        T_range (np.array): Array of time to maturity values to evaluate.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Greek calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Greeks for calls and puts over the stock price range:
              {'K': K,
                'S0': S0,
                'r': r,
                'ttm': T_range,
                'sigma': sigma,
                'greek': greek, 
                'greek_call': greek_call_list, 
                'greek_put': greek_put_list}.
    """

    # Fetch the greek from the mapping
    compute_greek = greeks_mapping.get(greek)
    if greek is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    greek_call_list = []
    greek_put_list = []

    for T in T_range:
        # Compute Delta for each stock price
        greeks = compute_greek(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        greek_call_list.append(greeks[str(greek) + '_call'])
        greek_put_list.append(greeks[str(greek) + '_put'])

    return {
        'K': K,
        'S0': S0,
        'r': r,
        'ttm': T_range,
        'sigma': sigma,
        'greek': greek, 
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_ttm(results):
    """
    Plot any greek (call and put) as a function of stock price (S0) using Plotly.
    """

    # unpack results in a better way 
    K = results['K']
    S0 = results['S0']
    T_range = results['ttm']
    r = results['r']
    sigma = results['sigma']
    greek = results['greek']

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Greek
    fig.add_trace(go.Scatter(
        x=results['ttm'],
        y=results[str(greek) + '_call'],
        mode='lines+markers',
        name='Call ' + str(greek).capitalize()
    ))

    # Add trace for Put Greek
    fig.add_trace(go.Scatter(
        x=results['ttm'],
        y=results[str(greek) + '_put'],
        mode='lines+markers',
        name='Put ' + str(greek).capitalize()
    ))

    # Add horizontal line at y=0
    fig.add_trace(go.Scatter(
        x=[min(T_range), max(T_range)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False  # Do not show in legend
    ))

    # Display input variables as annotations
    input_text = f"S0={S0:.2f}, K={K:.2f}, r={r:.2%}, σ={sigma:.2f}"
    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=input_text,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    # Update layout
    fig.update_layout(
        title=str(greek).capitalize() + " vs TTM",
        xaxis_title="Time to maturity (T)",
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


def greek_vs_volatility(Z: np.ndarray, 
                S0: float, 
                K: float, 
                T: float,
                r: float, 
                sigma_range: np.ndarray, 
                h: float, 
                exotic_type: str,
                greek: str, 
                n_simulations: int = 100000,
                **kwargs) -> dict:
    """
    Compute any greek (call and put) as a function of stock price (S0).

    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Stock price.
        K (float): Strike price.
        T float): Time to maturity.
        r (float): Risk-free rate.
        sigma_range (np.array): Array of volatilities to maturity values to evaluate.
        h (float): Small increment for Greek calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        greek (str): The greek we want to compute.
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).

    Returns:
        dict: Greeks for calls and puts over the stock price range:
              {'sigma': sigma_range,
                'S0': S0,
                'K': K,
                'r': r,
                'T': T,
                'greek': greek, 
                'greek_call': greek_call_list, 
                'greek_put': greek_put_list}.
    """

    # Fetch the greek from the mapping
    compute_greek = greeks_mapping.get(greek)
    if greek is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")
    
    greek_call_list = []
    greek_put_list = []

    for sigma in sigma_range:
        # Compute Delta for each stock price
        greeks = compute_greek(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        greek_call_list.append(greeks[str(greek) + '_call'])
        greek_put_list.append(greeks[str(greek) + '_put'])


    return {
        'sigma': sigma_range,
        'S0': S0,
        'K': K,
        'r': r,
        'T': T,
        'greek': greek, 
        str(greek) + '_call': greek_call_list,
        str(greek) + '_put': greek_put_list
    }


def plot_greek_vs_volatility(results):
    """
    Plot any greek (call and put) as a function of stock price (S0) using Plotly.
    """

    # unpack results in a better way 
    K = results['K']
    S0 = results['S0']
    T = results['T']
    r = results['r']
    sigma_range = results['sigma']
    greek = results['greek']

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Greek
    fig.add_trace(go.Scatter(
        x=results['sigma'],
        y=results[str(greek) + '_call'],
        mode='lines+markers',
        name='Call ' + str(greek).capitalize()
    ))

    # Add trace for Put Greek
    fig.add_trace(go.Scatter(
        x=results['sigma'],
        y=results[str(greek) + '_put'],
        mode='lines+markers',
        name='Put ' + str(greek).capitalize()
    ))

    # Add horizontal line at y=0
    fig.add_trace(go.Scatter(
        x=[min(sigma_range), max(sigma_range)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        showlegend=False  # Do not show in legend
    ))

    # Display input variables as annotations
    input_text = f"S0={S0:.2f}, T={T:.2f}, K={K:.2f}, r={r:.2%}"
    fig.add_annotation(
        x=0.5,
        y=-0.25,
        xref="paper",
        yref="paper",
        text=input_text,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    # Update layout
    fig.update_layout(
        title=str(greek).capitalize() + " vs Volatility",
        xaxis_title="Volatility (σ)",
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
    sigma_range = np.array([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8])  # Volatility range (as ndarray)
    sigma = 0.1
    n_simulations = 100000
    Z = np.random.standard_normal((n_simulations, 252))  # Precomputed random normals
    K = 100  # Strike price
    K_range = np.linspace(50, 150, 20)  # Strike price range (as ndarray)
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    h = 0.01  # Small increment for Delta calculation
    exotic_type = "asian"  # Exotic option type
    greek = 'delta'


    T_range = np.linspace(0.1, 2, 20)  # Time to Maturity from 0.1 to 1 years

    print(datetime.datetime.now())

    results = greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, greek)
    fig = plot_greek_vs_stock_price(results)
    fig.show()

    results = greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, greek)
    fig = plot_greek_vs_strike_price(results)
    fig.show()

    results = greek_vs_ttm(Z, S0, K, T_range, r, sigma, h, exotic_type, greek)
    fig = plot_greek_vs_ttm(results)
    fig.show()

    results = greek_vs_volatility(Z, S0, K, T, r, sigma_range, h, exotic_type, greek)
    fig = plot_greek_vs_volatility(results)
    fig.show()

    print(datetime.datetime.now())
    print('------------------')
