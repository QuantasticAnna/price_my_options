from dash import callback, Output, Input, State
import plotly.graph_objects as go
import numpy as np
import joblib

def setup_simulation_callbacks(app, monte_carlo_simulations, plotters, cyborg_template, exotic_option_types):
    """
    Sets up simulation callbacks for multiple exotic option types dynamically.

    Parameters:
        app (Dash): The Dash app instance.
        monte_carlo_simulations (function): Function to generate Monte Carlo simulations.
        plotters (dict): A dictionary mapping exotic option types to their respective plotter functions.
                         Example: {"asian": plotter_asian, "lookback": plotter_lookback}
        cyborg_template (str): Plotly template for styling plots.
        exotic_option_types (list): List of exotic option types (e.g., ["asian", "lookback"]).
    """
    # Define outputs dynamically based on exotic option types
    outputs = [Output(f"plot_first_n_simulations_{exotic}", "figure") for exotic in exotic_option_types]

    # Define inputs dynamically for the "Update Parameters" button
    inputs = [Input(f"button_update_params_{exotic}", "n_clicks") for exotic in exotic_option_types]

    @app.callback(
        outputs,
        inputs,
        [State(f"input_S0_{exotic}", "value") for exotic in exotic_option_types] +
        [State(f"input_K_{exotic}", "value") for exotic in exotic_option_types] +
        [State(f"input_T_{exotic}", "value") for exotic in exotic_option_types] +
        [State(f"input_r_{exotic}", "value") for exotic in exotic_option_types] +
        [State(f"input_sigma_{exotic}", "value") for exotic in exotic_option_types],
    )
    def update_simulation_plots(*args):
        """
        Callback to update simulation plots for multiple exotic options.

        Parameters:
            args: Dynamically passed inputs and states.

        Returns:
            tuple: Figures for each exotic option type.
        """
        n_exotics = len(exotic_option_types)
        n_clicks = args[:n_exotics]  # Extract the button clicks
        states = args[n_exotics:]  # Extract the state values

        # Split the states by exotic option type
        split_states = [states[i::n_exotics] for i in range(n_exotics)]
        figures = []

        for exotic, clicks, state in zip(exotic_option_types, n_clicks, split_states):

            Z_precomputed = joblib.load("Z_precomputed.joblib")

            if clicks > 0 and Z_precomputed is not None:
                S0, K, T, r, sigma = state
                Z = np.array(Z_precomputed)  # Convert Z back to NumPy array
                S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=100000)

                # Generate plot using the corresponding plotter
                plotter = plotters.get(exotic)
                if plotter:
                    fig = plotter(S, n_sim_to_plot=10)
                    fig.add_hline(
                        y=K,
                        line=dict(color="white", width=2, dash="dash"),
                        annotation_text=f"Strike Price (K={K})",
                        annotation_position="bottom right",
                    )
                    figures.append(fig)
                else:
                    figures.append(empty_figure(cyborg_template))
            else:
                figures.append(empty_figure(cyborg_template))

        return tuple(figures)

    def empty_figure(template):
        """Creates an empty placeholder figure."""
        fig = go.Figure()
        fig.update_layout(
            title="First Monte Carlo Simulations of Stock Prices, with avg",
            xaxis_title="Time Steps",
            yaxis_title="Stock Price",
            template=template,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        return fig
