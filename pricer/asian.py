import numpy as np
import plotly.graph_objects as go
from pricer.monte_carlo import monte_carlo_simulations, monte_carlo_simulations_heston
from custom_templates import cyborg_template
import datetime

def pricer_asian(S, K, T, r): #! at some point, add the option arithmetic / geometric for mean computation 
    """
    Asian option pricer using Monte Carlo simulations.
    
    Parameters:
        S (np.ndarray): Array of Monte Carlo simulation paths.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.

    Returns:
        dict: Price of the Asian call, Price of the Asian put
    """

    discount_factor = np.exp(-r * T)  # Discount factor for risk-free rate

    # Calculate average price for each path
    average_price = np.mean(S[:, 1:], axis=1)

    # Calculate payoffs
    payoff_call = np.maximum(average_price - K, 0)
    payoff_put = np.maximum(K - average_price, 0)

    # Return discounted payoff
    price_call = discount_factor * np.mean(payoff_call)
    price_put = discount_factor * np.mean(payoff_put)

    return {'price_call': price_call, 
            'price_put': price_put}


def plotter_asian(S: np.ndarray, 
                  n_sim_to_plot=10) -> go.Figure:
    """
    Add to the base plot of Monte Carlo paths the specificities for Asian options: 
    The average of each trajectory.
    Note: We want to plot the avg line in the same color than the path line, 
    that's why we access the colors used in the base plot. 

    Parameters:
        S (np.ndarray): Simulated price paths (matrix of shape [n_simulations, n_steps+1]).
        n_sim_to_plot (int): Number of simulations to plot.

    Returns:
        go.Figure: A Plotly figure object.
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

    # Access the colors used in the base plot
    colors = [trace.line.color for trace in fig.data if trace.mode == 'lines']

    # Add traces for the first n_simulations
    for i in range(n_sim_to_plot):
        avg_price = np.mean(S[i, :])
        color = colors[i] if i < len(colors) else 'blue'  # Fallback to 'blue'

        # Horizontal line for the average price
        fig.add_trace(go.Scatter(
            x=[0, S.shape[1] - 1],
            y=[avg_price, avg_price],
            mode='lines',
            line=dict(color=color, dash='dot', width=2),
            name=f'Avg Simulation {i+1}',
            showlegend=True
        ))

    # Layout settings
    fig.update_layout(
        title=f"First {n_sim_to_plot} Monte Carlo Simulations of Stock Prices, with avg",
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
    n_simulations = 100000
    n_sim_to_plot = 10

    Z = np.random.standard_normal((n_simulations, 252))

    # Simulate Asian option payoffs
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)

    S_heston = monte_carlo_simulations_heston(
        Z, S0=100, T=1, r=0.05, v0=0.04, 
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7
    )

    print(datetime.datetime.now())

    # Plot 
    # Get the base plot from the other function
    # fig = plotter_first_n_simulations(S, n_sim_to_plot)
    fig_asian = plotter_asian(S, n_sim_to_plot=10)
    fig_asian.show()

    prices_heston = pricer_asian(S_heston, 100, T, r)
    fig_asian_heston = plotter_asian(S_heston, n_sim_to_plot=10)
    
    fig_asian_heston.show()

    print(datetime.datetime.now())

    print('-----------------')