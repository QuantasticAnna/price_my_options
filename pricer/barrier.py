import numpy as np
import plotly.graph_objects as go
from pricer.monte_carlo import plotter_first_n_simulations, monte_carlo_simulations
from custom_templates import cyborg_template

def pricer_barrier(S: np.ndarray, K: float, T: float, r: float, B_call: float, B_put: float) -> dict:
    """
    Barrier option pricer using Monte Carlo simulations.
    
    Parameters:
        S (np.ndarray): Array of Monte Carlo simulation paths (matrix of shape [n_simulations, n_steps+1]).
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        B_call (float): Barrier for Down-and-Out Call.
        B_put (float): Barrier for Up-and-Out Put.

    Returns:
        dict: Prices of the Down-and-Out Call and Up-and-Out Put.
    """
    discount_factor = np.exp(-r * T)  # Discount factor for risk-free rate

    # Down-and-Out Call: Valid paths where all S_t > B_call
    valid_call_paths = np.all(S > B_call, axis=1)
    payoff_call = np.maximum(S[:, -1] - K, 0) * valid_call_paths

    # Up-and-Out Put: Valid paths where all S_t < B_put
    valid_put_paths = np.all(S < B_put, axis=1)
    payoff_put = np.maximum(K - S[:, -1], 0) * valid_put_paths

    # Compute discounted prices
    price_down_and_out_call = discount_factor * np.mean(payoff_call)
    price_up_and_out_put = discount_factor * np.mean(payoff_put)

    return {
        'price_call': price_down_and_out_call,
        'price_put': price_up_and_out_put
    }



def plotter_barrier(S: np.ndarray, B_call: float, B_put: float, n_sim_to_plot=10) -> tuple[go.Figure, go.Figure]:
    """
    Plot Monte Carlo paths for Barrier options: Down-and-Out Call and Up-and-Out Put.
    Invalid paths (crossing the barrier) are dashed, and a red dot marks the first crossing point.

    Parameters:
        S (np.ndarray): Simulated price paths (matrix of shape [n_simulations, n_steps+1]).
        B_call (float): Barrier for Down-and-Out Call.
        B_put (float): Barrier for Up-and-Out Put.
        n_sim_to_plot (int): Number of simulations to plot.

    Returns:
        tuple[go.Figure, go.Figure]: Two Plotly figures, one for Down-and-Out Call and one for Up-and-Out Put.
    """

    # Plot for Down-and-Out Call
    fig_down_and_out_call = go.Figure()
    time_steps = np.arange(S.shape[1])

    for i in range(n_sim_to_plot):
        valid_call = np.all(S[i, :] > B_call)

        # Solid for valid paths, dashed for invalid
        line_dash = "solid" if valid_call else "dash"
        fig_down_and_out_call.add_trace(go.Scatter(
            x=time_steps,
            y=S[i, :],
            mode="lines",
            line=dict(dash=line_dash),
            name=f"Simulation {i+1}",
            showlegend=True
        ))

        # Mark the first crossing for Down-and-Out Call
        if not valid_call:
            crossing_index = np.argmax(S[i, :] <= B_call)  # First crossing where S_t <= B_call
            fig_down_and_out_call.add_trace(go.Scatter(
                x=[time_steps[crossing_index]],
                y=[S[i, crossing_index]],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name=f'Barrier Crossing (Sim {i+1})',
                showlegend=True
            ))

    # Add the barrier line for Down-and-Out Call
    fig_down_and_out_call.add_trace(go.Scatter(
        x=[0, S.shape[1] - 1],
        y=[B_call, B_call],
        mode="lines",
        line=dict(color="green", width=2, dash="dot"),
        name=f"Barrier B_call = {B_call}"
    ))
    fig_down_and_out_call.update_layout(
        title="Down-and-Out Call - Monte Carlo Simulations",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template=cyborg_template
    )

    # Plot for Up-and-Out Put
    fig_up_and_out_put = go.Figure()

    for i in range(n_sim_to_plot):
        valid_put = np.all(S[i, :] < B_put)

        # Solid for valid paths, dashed for invalid
        line_dash = "solid" if valid_put else "dash"
        fig_up_and_out_put.add_trace(go.Scatter(
            x=time_steps,
            y=S[i, :],
            mode="lines",
            line=dict(dash=line_dash),
            name=f"Simulation {i+1}",
            showlegend=True
        ))

        # Mark the first crossing for Up-and-Out Put
        if not valid_put:
            crossing_index = np.argmax(S[i, :] >= B_put)  # First crossing where S_t >= B_put
            fig_up_and_out_put.add_trace(go.Scatter(
                x=[time_steps[crossing_index]],
                y=[S[i, crossing_index]],
                mode='markers',
                marker=dict(color='red', size=8, symbol='circle'),
                name=f'Barrier Crossing (Sim {i+1})',
                showlegend=True
            ))

    # Add the barrier line for Up-and-Out Put
    fig_up_and_out_put.add_trace(go.Scatter(
        x=[0, S.shape[1] - 1],
        y=[B_put, B_put],
        mode="lines",
        line=dict(color="green", width=2, dash="dot"),
        name=f"Barrier B_put = {B_put}"
    ))
    fig_up_and_out_put.update_layout(
        title="Up-and-Out Put - Monte Carlo Simulations",
        xaxis_title="Time Steps",
        yaxis_title="Stock Price",
        template=cyborg_template
    )

    return fig_down_and_out_call, fig_up_and_out_put




if __name__ == "__main__":
    # Parameters
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    n_simulations = 100000
    n_sim_to_plot = 10
    B_call = 90  # Barrier for Down-and-Out Call
    B_put = 110  # Barrier for Up-and-Out Put

    Z = np.random.standard_normal((n_simulations, 252))

    # Simulate Asian option payoffs
    S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations)

    results = pricer_barrier(S, K, T, r, B_call, B_put)

    fig_down_and_out_call, fig_up_and_out_put = plotter_barrier(S, B_call, B_put)

    # Display the plots
    fig_down_and_out_call.show()
    fig_up_and_out_put.show()

    print(results)