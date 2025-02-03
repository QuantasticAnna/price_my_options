import numpy as np
import plotly.graph_objects as go
from pricer_plotter.monte_carlo import monte_carlo_simulations
from pricer_plotter.custom_templates import cyborg_template


def pricer_european(S: np.ndarray, K: float, T: float, r: float) -> dict:
    """
    European option pricer using Monte Carlo simulations.

    Parameters:
        S (np.ndarray): Array of Monte Carlo simulation paths.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.

    Returns:
        dict: Price of the European call and put options.
    """
    discount_factor = np.exp(-r * T)  # Discount factor for risk-free rate

    # Get final prices (S_T) from simulation
    S_T = S[:, -1]

    # Calculate payoffs
    payoff_call = np.maximum(S_T - K, 0)
    payoff_put = np.maximum(K - S_T, 0)

    # Calculate discounted average payoffs
    price_call = discount_factor * np.mean(payoff_call)
    price_put = discount_factor * np.mean(payoff_put)

    return {
        'price_call': price_call,
        'price_put': price_put
    }


def plotter_european(S: np.ndarray, n_sim_to_plot: int = 10) -> go.Figure:
    """
    Plot the Monte Carlo paths and the final payoffs for European options.

    Parameters:
        S (np.ndarray): Simulated price paths (matrix of shape [n_simulations, n_steps+1]).
        K (float): Strike price.
        n_sim_to_plot (int): Number of simulations to plot.

    Returns:
        go.Figure: A Plotly figure object.
    """
    fig = go.Figure()

    # Add traces for the first n_simulations
    for i in range(n_sim_to_plot):
        time_steps = np.arange(S.shape[1])
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=S[i, :],
            mode='lines',
            name=f'Simulation {i+1}',
            showlegend=True
        ))

    # Final price line and strike price line
    final_prices = S[:n_sim_to_plot, -1]
    for i, final_price in enumerate(final_prices):
        fig.add_trace(go.Scatter(
            x=[S.shape[1] - 1],
            y=[final_price],
            mode='markers',
            marker=dict(color='blue', size=8),
            name=f'Final Price (Sim {i+1})',
            showlegend=False
        ))

    # Layout settings
    fig.update_layout(
        title=f"Monte Carlo Simulations for European Option (First {n_sim_to_plot} Paths)",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template=cyborg_template
    )

    return fig


if __name__ == '__main__':
    # Parameters
    S0 = 100
    T = 1
    r = 0.05
    sigma = 0.2
    K = 100  # Strike price
    n_simulations = 100000
    n_sim_to_plot = 10

    Z = np.random.standard_normal((n_simulations, 252))

    # Simulate price paths
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)

    # Price European options
    prices = pricer_european(S, K, T, r)
    print(f"European Call Price: {prices['price_call']}")
    print(f"European Put Price: {prices['price_put']}")

    # Plot European options
    fig_european = plotter_european(S, n_sim_to_plot)
    fig_european.show()
