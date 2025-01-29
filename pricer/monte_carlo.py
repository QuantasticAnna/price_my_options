import numpy as np
import plotly.graph_objects as go
import datetime


def monte_carlo_simulations(Z: np.ndarray, # Z is an input of monte carlo, so we define it once, for all the application 
                            S0: float, 
                            T: float, 
                            r: float, 
                            sigma: float, 
                            n_simulations: int = 100000
                            ) -> tuple[np.ndarray, np.ndarray]:

    """
    Monte Carlo simulations generator.

    Parameters:
        S0 (float): Initial stock price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        n_simulations (int, optional): Number of Monte Carlo simulations. Default is 100000.
        h (float, optional): Increment for Delta calculation. Default is 0.01.

    Note: We generate one array with the stock price starting at S0, 
        and a second array with the stock price starting at S0 + h.
        The seconda rray will be used to compute greeks, using the finite difference method.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays of simulated paths: one starting at S0 and the other at S0 + h.
    """
    # Infer number of time steps
    n_steps = int(np.ceil(252 * T))  # Assume 252 trading days per year
    dt = T / n_steps  # Time step size

    # Ensure Z has sufficient columns
    if Z.shape[1] < n_steps:
        raise ValueError(f"Z array has insufficient columns: {Z.shape[1]} < {n_steps}")

    # Simulations for S0
    S = np.zeros((n_simulations, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

    return S


def monte_carlo_simulations_heston(Z: np.ndarray, 
                                   S0: float, 
                                   T: float, 
                                   r: float, 
                                   v0: float, 
                                   kappa: float, 
                                   theta: float, 
                                   xi: float, 
                                   rho: float, 
                                   n_simulations: int = 100000) -> np.ndarray:
    """
    Monte Carlo simulation for the Heston stochastic volatility model.

    Parameters:
        Z (np.ndarray): Pre-generated standard normal random variables for Monte Carlo.
        S0 (float): Initial stock price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        v0 (float): Initial variance (sigma^2).
        kappa (float): Speed of mean reversion for variance.
        theta (float): Long-run mean variance.
        xi (float): Volatility of volatility.
        rho (float): Correlation between stock price and variance.
        n_simulations (int, optional): Number of Monte Carlo simulations. Default is 100000.

    Returns:
        np.ndarray: Simulated paths of stock prices.
    """

    # Time step settings
    n_steps = 252  # Assume 252 trading days per year
    dt = T / n_steps  # Time step size

    # Ensure Z has correct shape
    if Z.shape != (n_simulations, n_steps):
        raise ValueError(f"Z must have shape ({n_simulations}, {n_steps}) but got {Z.shape}")

    # Simulated paths for S (stock price) and v (variance)
    S = np.zeros((n_simulations, n_steps + 1))
    v = np.zeros((n_simulations, n_steps + 1))
    
    # Initial values
    S[:, 0] = S0
    v[:, 0] = v0

    # Generate second correlated normal variable for variance using Cholesky decomposition
    Z_V = np.random.standard_normal((n_simulations, n_steps))
    W_S = Z
    W_V = rho * Z + np.sqrt(1 - rho**2) * Z_V  # Correlated Wiener process

    for t in range(1, n_steps + 1):
        # Ensure variance remains positive using max(v_t, 0)
        v[:, t] = np.maximum(v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + xi * np.sqrt(v[:, t-1] * dt) * W_V[:, t-1], 0)

        # Stock price evolution
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t-1]) * dt + np.sqrt(v[:, t-1] * dt) * W_S[:, t-1])

    return S

def plotter_first_n_simulations(S: np.ndarray, 
                                 n_sim_to_plot: int = 10) -> go.Figure:
    """
    Plot the first n_simulations of the stock price matrix S using Plotly.

    Parameters:
        S (np.ndarray): Simulated price paths (matrix of shape [n_simulations, n_steps+1]).
        n_sim_to_plot (int, optional): Number of simulations to plot. Default is 10.
    """
    fig = go.Figure()

    # Add traces for the first n_simulations
    for i in range(n_sim_to_plot):
        time_steps = np.arange(S.shape[1])

        # Line for the simulation path
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=S[i, :],
            mode='lines',
            name=f'Simulation {i+1}',
            showlegend=True
        ))

        # Marker for the maximum price

    # Layout settings
    fig.update_layout(
        title=f"First {n_sim_to_plot} Monte Carlo Simulations of Stock Prices",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':

    # Parameters
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    n_simulations = 100000
    h = 0.01

    Z = np.random.standard_normal((n_simulations, 252))

    # Simulate Asian option payoffs
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)    

    # Plots
    fig_S = plotter_first_n_simulations(S, 10)

    fig_S.show()


    S_paths = monte_carlo_simulations_heston(
        Z, S0=100, T=1, r=0.05, v0=0.04, 
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
    )