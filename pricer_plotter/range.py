import numpy as np
import plotly.graph_objects as go
from pricer_plotter.monte_carlo import monte_carlo_simulations

def pricer_range(S, K, T, r, lower_bound, upper_bound):
    """
    Range option pricer using Monte Carlo simulations.
    
    Parameters:
        S (np.ndarray): Array of Monte Carlo simulation paths.
        K (float): Payout multiplier.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        lower_bound (float): Lower price bound of the range.
        upper_bound (float): Upper price bound of the range.

    Returns:
        float: Price of the range option.
    """
    discount_factor = np.exp(-r * T)  # Discount factor for risk-free rate

    # Count the number of time steps each path stays in the range
    time_in_range = np.sum((S[:, 1:] >= lower_bound) & (S[:, 1:] <= upper_bound), axis=1)

    # Calculate the proportion of time spent in range
    proportion_in_range = time_in_range / S[:, 1:].shape[1]

    # Calculate payoffs
    payoff = K * proportion_in_range

    # Return discounted payoff
    price = discount_factor * np.mean(payoff)

    return price


def plotter_range():
    pass


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

    # Price the range option
    lower_bound = 90
    upper_bound = 110
    K = 10

    range_option_price = pricer_range(S, K, T, r, lower_bound, upper_bound)
    print(f"Range Option Price: {range_option_price:.2f}")
