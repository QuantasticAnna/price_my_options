import numpy as np
from pricer.asian import pricer_asian
from pricer.monte_carlo import plotter_first_n_simulations, monte_carlo_simulations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from config import pricer_mapping

# TODO: 
    # compute_delta
    # delta_vs_stock_price
    # delta_vs_ttm
    # delta_vs_vola
    # plot_delta_vs_stock_price
    # plot_3d_delta_vs_diff_implied_vol
    # plot_3d_delta_over_time

# def compute_delta(Z: np.ndarray, 
#                   S0: float,
#                   K: float, 
#                   T: float, 
#                   r: float, 
#                   sigma: float, 
#                   h: float,
#                   n_simulations: int = 100000) -> float:
#     """

#     TO UPDATE

#     Compute Delta for an Asian option

#     Parameters:

#         K (float): Strike price.
#         T (float): Time to maturity.
#         r (float): Risk-free rate.
#         h (float): Small increment for Delta calculation.

#     Returns:
#         float: Estimated Delta.
#     """

#     S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
#     S_h = monte_carlo_simulations(Z, S0 + h, T, r, sigma, n_simulations)

#     # Note: here, when other pricers will be implement, there will be a if type == 'asian', elif 'barrier' etc

#     # Price Asian options for both S and S_h
#     prices_S = pricer_asian(S, K, T, r)
#     prices_S_h = pricer_asian(S_h, K, T, r)

#     # Extract prices for call options
#     price_call_S = prices_S['price_call']
#     price_call_S_h = prices_S_h['price_call']

#     # Compute Delta via finite difference
#     delta_call = (price_call_S_h - price_call_S) / h

#     # Extract prices for put options
#     price_put_S = prices_S['price_put']
#     price_put_S_h = prices_S_h['price_put']

#     # Compute Delta via finite difference
#     delta_put = (price_put_S_h - price_put_S) / h


#     return {'delta_call': delta_call,
#             'delta_put': delta_put} 

def compute_delta(Z: np.ndarray, 
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
    Compute Delta for an exotic option using the appropriate pricer.
    
    Parameters:
        Z (np.ndarray): Precomputed random normals for Monte Carlo simulation.
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Volatility.
        h (float): Small increment for Delta calculation.
        exotic_type (str): Type of exotic option (e.g., "asian", "barrier").
        n_simulations (int): Number of Monte Carlo simulations. Default is 100000.
        **kwargs: Additional parameters for specific exotic options (e.g., "barrier" for barrier options).
    
    Returns:
        dict: Delta for call and put options.
    """
    # Fetch the pricer from the mapping
    pricer = pricer_mapping.get(exotic_type)
    if pricer is None:
        raise ValueError(f"Unsupported exotic_type: {exotic_type}")

    # Simulate prices
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)
    S_h = monte_carlo_simulations(Z, S0 + h, T, r, sigma, n_simulations)

    # Price options using the appropriate pricer
    prices_S = pricer(S, K, T, r, **kwargs)
    prices_S_h = pricer(S_h, K, T, r, **kwargs)

    # Compute Delta for call and put
    delta_call = (prices_S_h['price_call'] - prices_S['price_call']) / h
    delta_put = (prices_S_h['price_put'] - prices_S['price_put']) / h

    return {'delta_call': delta_call, 'delta_put': delta_put}



def delta_vs_stock_price(Z: np.ndarray, 
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
    Compute Delta (call and put) as a function of stock price (S0).

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

    Returns:
        dict: Deltas for calls and puts over the stock price range:
              {'stock_price': S0_range, 'delta_call': delta_call_list, 'delta_put': delta_put_list}.
    """
    delta_call_list = []
    delta_put_list = []

    for S0 in S0_range:
        # Compute Delta for each stock price
        deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        delta_call_list.append(deltas['delta_call'])
        delta_put_list.append(deltas['delta_put'])

    return {
        'stock_price': S0_range,
        'delta_call': delta_call_list,
        'delta_put': delta_put_list
    }


def delta_vs_strike_price(Z: np.ndarray, 
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
    Compute Delta (call and put) as a function of STRIKE price for a fixed stock price using Plotly.

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

    Returns:
        dict: Deltas for calls and puts over the stock price range:
              {'stock_price': S0_range, 'delta_call': delta_call_list, 'delta_put': delta_put_list}.
    """
    delta_call_list = []
    delta_put_list = []

    for K in K_range:
        # Compute Delta for each stock price
        deltas = compute_delta(Z, S0, K, T, r, sigma, h, exotic_type, n_simulations=n_simulations, **kwargs)
        delta_call_list.append(deltas['delta_call'])
        delta_put_list.append(deltas['delta_put'])

    return {
        'strike_price': K_range,
        'delta_call': delta_call_list,
        'delta_put': delta_put_list
    }



def plot_delta_vs_stock_price(Z: np.ndarray, 
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
    Plot Delta (call and put) as a function of stock price (S0) using Plotly.

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
    # Compute Delta vs Stock Price
    results = delta_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic_type, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Delta
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['delta_call'],
        mode='lines+markers',
        name='Call Delta'
    ))

    # Add trace for Put Delta
    fig.add_trace(go.Scatter(
        x=results['stock_price'],
        y=results['delta_put'],
        mode='lines+markers',
        name='Put Delta'
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
        title="Delta vs Stock Price",
        xaxis_title="Stock Price (S0)",
        yaxis_title="Delta",
        legend=dict(title="Option Type"),
        template="plotly_white"
    )

    # Show the figure
    fig.show()


def plot_delta_vs_strike_price(Z: np.ndarray, 
                              S0: np.ndarray, 
                              K_range: float, 
                              T: float, 
                              r: float, 
                              sigma: float, 
                              h: float, 
                              exotic_type: str,
                              n_simulations: int = 100000,
                              **kwargs):
    """
    Plot Delta (call and put) as a function of STRIKE price for a fixed stock price using Plotly.

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
    # Compute Delta vs Stock Price
    results = delta_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, n_simulations, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Add trace for Call Delta
    fig.add_trace(go.Scatter(
        x=results['strike_price'],
        y=results['delta_call'],
        mode='lines+markers',
        name='Call Delta'
    ))

    # Add trace for Put Delta
    fig.add_trace(go.Scatter(
        x=results['strike_price'],
        y=results['delta_put'],
        mode='lines+markers',
        name='Put Delta'
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
        title="Delta vs Strike Price",
        xaxis_title="Strike Price (K)",
        yaxis_title="Delta",
        legend=dict(title="Option Type"),
        template="plotly_white"
    )

    # Show the figure
    fig.show()


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
    delta_matrix_call = []
    delta_matrix_put = []

    # Iterate over volatilities
    for sigma in volatilities:
        # Compute Delta for each volatility
        results = delta_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic_type, n_simulations, **kwargs)
        delta_matrix_call.append(results['delta_call'])
        delta_matrix_put.append(results['delta_put'])

    return {
        'strike_price': K_range,
        'volatility': volatilities,
        'delta_matrix_call': np.array(delta_matrix_call),
        'delta_matrix_put': np.array(delta_matrix_put)
    }


def plot_3d_delta_vs_strike_price_for_multiple_volatility(results: dict, option_type: str = "call"):
    """
    Plot a 3D surface of Delta vs Stock Price and Volatility.

    Parameters:
        results (dict): Output from `delta_vs_stock_price_for_multiple_volatility`.
        option_type (str): Either "call" or "put" to specify which Delta to plot.
    """
    # Extract data
    strike_prices = results['strike_price']
    volatilities = results['volatility']
    delta_matrix = results['delta_matrix_call'] if option_type == "call" else results['delta_matrix_put']

    # Create the 3D surface plot
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=delta_matrix,
        x=strike_prices,  # Stock prices
        y=volatilities,  # Volatility
        colorscale="Viridis",
        colorbar=dict(title="Delta"),
    ))

    # Update layout
    fig.update_layout(
        title=f"3D Delta Surface ({option_type.capitalize()} Option)",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Volatility (σ)",
            zaxis_title="Delta",
        ),
        template="plotly_white"
    )

    # Show the figure
    fig.show()



def delta_vs_ttm():
    pass

def delta_vs_vol():
    pass

# if __name__ == "__main__":
#     # Parameters
#     S0 = 100  # Initial stock price
#     T = 1  # Time to maturity (1 year)
#     r = 0.05  # Risk-free rate
#     sigma = 0.2  # Volatility
#     K = 100  # Strike price
#     h = 0.01  # Small shift for Delta
#     n_simulations = 100000


#     # Define stock price range
#     S0_range = np.linspace(50, 150, 20)  # Stock prices from 50 to 150

#     # Generate Z once
#     Z = np.random.standard_normal((n_simulations, 252))

#     # Plot Delta vs Stock Price

#     plot_delta_vs_stock_price(
#         Z=Z,
#         S0_range=S0_range,
#         K=100,
#         T=1,
#         r=0.05,
#         sigma=0.2,
#         h=0.01,
#         exotic_type="asian",
#         n_simulations=100000,
#         # barrier=120  # Barrier parameter
#     )

#     delta = compute_delta(
#     Z=Z,
#     S0=100,
#     K=100,
#     T=1,
#     r=0.05,
#     sigma=0.2,
#     h=0.01,
#     exotic_type="asian",
#     # barrier=120  # Additional parameter for barrier options

# )
    
#     print(delta)


import numpy as np
from plotly.graph_objects import Surface

if __name__ == "__main__":
    # Define parameters
    S0_range = np.linspace(50, 150, 20)  # Stock price range (as ndarray)
    S0 = 100
    volatilities = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Volatility range (as ndarray)
    sigma = 0.1
    Z = np.random.standard_normal((100000, 252))  # Precomputed random normals
    K = 100  # Strike price
    K_range = np.linspace(50, 150, 20)  # Strike price range (as ndarray)
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    h = 0.01  # Small increment for Delta calculation
    exotic_type = "asian"  # Exotic option type

    # plot_delta_vs_strike_price(Z, 
    #                           S0, 
    #                           K_range, 
    #                           T, 
    #                           r, 
    #                           sigma, 
    #                           h, 
    #                           exotic_type)
    
    # # Compute Delta matrix for multiple volatilities
    results = delta_vs_strike_price_for_multiple_volatility(
        Z=Z,
        S0=S0,
        volatilities=volatilities,
        K_range=K_range,
        T=T,
        r=r,
        h=h,
        exotic_type=exotic_type,
        n_simulations=100000
    )

    # # Plot 3D Delta vs Stock Price for Call Options
    plot_3d_delta_vs_strike_price_for_multiple_volatility(results, option_type="call")

    # explanantion: its beacuse the graphs on the book, the xaxis is strike price, not stock price 
    #! In dash, maye instead of making a 3d plot, i can make a 2d graph, with already the values computed for different vol, 
    # and change it with a slider  