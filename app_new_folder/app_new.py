from dash import Dash, Input, Output, html, dcc, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import dash_mantine_components as dmc
import joblib
from pricer.asian import plotter_asian  # should be in ascript plotter
from pricer.lookback import plotter_lookback
from pricer.monte_carlo import monte_carlo_simulations
import plotly.graph_objects as go
from app_new_folder.components import generate_main_div, empty_fig  # Import reusable components
from constants import H, S0_RANGE, K_RANGE
from greeks.delta import compute_delta
from greeks.gamma import compute_gamma
from greeks.vega import compute_vega
from greeks.theta import compute_theta
from greeks.rho import compute_rho
from greeks.greeks_functions import plot_greek_vs_stock_price, plot_greek_vs_strike_price

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY, "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.9.1/font/bootstrap-icons.min.css"], )

app.title = "Price My Options NEW"

# Load precomputed Z
Z_precomputed = joblib.load("Z_precomputed.joblib")

# Exotic options dynamically retrieved from menu_bar
EXOTIC_OPTION_TYPES = ["asian", "lookback"] 

GREEKS = ["delta", "gamma", "theta", "vega", "rho"]

# Dictionary of exotic option types and their corresponding plotters
PLOTTERS = {
    "asian": plotter_asian,
    "lookback": plotter_lookback,
}

# Menu bar for selecting exotic options
menu_bar = html.Div([
    dmc.SegmentedControl(
        id="menu_bar",
        value="asian",
        fullWidth=True,
        data=[
            {"value": "asian", "label": "Asian"},
            {"value": "lookback", "label": "Lookback"},
            #{"value": "value3", "label": "Label 3"},
        ]
    )
])

# Generate divs for exotic options
div_asian = generate_main_div("asian")
div_lookback = generate_main_div("lookback")
div3 = html.Div(html.H4("Placeholder for Value 3"), id="div_value3", hidden=True)

# Define the app layout
app.layout = html.Div([
    html.H1("Price My Options NEW", style={"textAlign": "center", "margin-top": "20px"}),
    menu_bar,
    div_asian,
    div_lookback,
    div3
], style = {'margin' : '30px'})

# Callback to toggle visibility of divs based on the menu bar selection
@app.callback(
    [Output('div_asian', 'hidden'),
     Output('div_lookback', 'hidden'),
     Output('div_value3', 'hidden')],
    [Input('menu_bar', 'value')]
)
def show_hidden_div(input_value):
    # Default all divs to hidden
    show_div_asian = True
    show_div_lookback = True
    show_div3 = True

    # Show only the selected div
    if input_value == 'asian':
        show_div_asian = False
    elif input_value == 'lookback':
        show_div_lookback = False
    elif input_value == 'value3':
        show_div3 = False

    return show_div_asian, show_div_lookback, show_div3



@app.callback(
    [Output(f"plot_first_n_simulations_{exotic}", "figure") for exotic in EXOTIC_OPTION_TYPES],
    [Input(f"button_update_params_{exotic}", "n_clicks") for exotic in EXOTIC_OPTION_TYPES],
    [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES],
)
def show_plot_first_n_simulations(*args):
    """
    Callback to generate and update simulation plots for multiple exotic options.

    Parameters:
        args: A combination of n_clicks and state values dynamically passed.

    Returns:
        tuple: Figures for each exotic option type.
    """
    # Separate button clicks and state values
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_clicks = args[:n_exotics]
    states = args[n_exotics:]

    # Split states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]
    figures = []

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):
        if clicks > 0 and Z_precomputed is not None:
            S0, K, T, r, sigma = state
            Z = np.array(Z_precomputed)  # Convert Z back to NumPy array
            S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=100000)

            # Get the appropriate plotter for the exotic option type
            plotter = PLOTTERS[exotic]
            fig = plotter(S, n_sim_to_plot=10)
            fig.add_hline(
                y=K,
                line=dict(color="white", width=2, dash="dash"),
                annotation_text=f"Strike Price (K={K})",
                annotation_position="bottom right",
            )
            figures.append(fig)
        else:
            figures.append(empty_fig) # empty fig

    return tuple(figures)



@app.callback(
    [
        Output(f"{greek}_{option_type}_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
        for option_type in ["call", "put"]
    ],
    [Input(f"button_update_params_{exotic}", "n_clicks") for exotic in EXOTIC_OPTION_TYPES],
    [
        State(f"input_S0_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value")
        for exotic in EXOTIC_OPTION_TYPES
    ],
)
def update_greeks(*args):
    """
    Callback to compute and display Greek values (Delta, Gamma, Theta, Vega, Rho)
    for multiple exotic options dynamically.

    Parameters:
        args: Dynamically passed inputs and states.

    Returns:
        tuple: Greek values for calls and puts for all exotic options.
    """

    # Separate button clicks and state values
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)
    n_clicks = args[:n_exotics]
    states = args[n_exotics:]

    # Split states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]
    results = []

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):
        if clicks > 0 and Z_precomputed is not None:
            S0, K, T, r, sigma = state
            h = H

            # Compute Greeks
            deltas = compute_delta(Z_precomputed, S0, K, T, r, sigma, h, exotic)
            gammas = compute_gamma(Z_precomputed, S0, K, T, r, sigma, h, exotic)
            thetas = compute_theta(Z_precomputed, S0, K, T, r, sigma, h, exotic)
            vegas = compute_vega(Z_precomputed, S0, K, T, r, sigma, h, exotic)
            rhos = compute_rho(Z_precomputed, S0, K, T, r, sigma, h, exotic)

            # Append results for call and put values
            results.extend([
                html.Div(f"{deltas['delta_call']:.2f}"),  # Delta Call
                html.Div(f"{deltas['delta_put']:.2f}"),   # Delta Put

                html.Div(f"{gammas['gamma_call']:.2f}"),  # Gamma Call
                html.Div(f"{gammas['gamma_put']:.2f}"),   # Gamma Put

                html.Div(f"{thetas['theta_call']:.2f}"),       # Theta Call
                html.Div(f"{thetas['theta_put']:.2f}"),        # Theta Put

                html.Div(f"{vegas['vega_call']:.2f}"),    # Vega Call
                html.Div(f"{vegas['vega_put']:.2f}"),     # Vega Put

                html.Div(f"{rhos['rho_call']:.2f}"),      # Rho Call
                html.Div(f"{rhos['rho_put']:.2f}"),       # Rho Put
            ])

        else:
            # Empty values for all Greeks (call and put)
            results.extend([html.Div('') for _ in range(n_greeks * 2)])

    return tuple(results) # TODO: in this callback, also ouptu the results in the greek table



@app.callback(
    [
        Output(f"store_plot_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_stock_price_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_plot_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] + [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ],
)
def update_greek_vs_stock_price_plots(*args):
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks : n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Group states by exotic type
    n_states_per_exotic = 5  # S0, K, T, r, sigma
    split_states = [
        states[i::n_exotics] for i in range(n_exotics)
    ]

    # Initialize updated plots with empty_fig by default
    updated_plots = [
        stored_data[i] if stored_data[i] is not None else empty_fig
        for i in range(len(stored_data))
    ]

    for exotic_index, (exotic, state_set) in enumerate(zip(EXOTIC_OPTION_TYPES, split_states)):
        S0, K, T, r, sigma = state_set

        h = H
        S0_range = S0_RANGE
        Z = Z_precomputed

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_button == f"button_compute_{greek}_vs_stock_price_{exotic}" and Z is not None:
                plot = plot_greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek)
                updated_plots[output_index] = plot

    return tuple(updated_plots)




@app.callback(
    [
        Output(f"plot_{greek}_vs_stock_price_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_plot_{greek}_vs_stock_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_stock_price_from_store(*stored_data):
    """
    Callback to render stored plots in their respective graphs.
    """
    # return tuple(stored_data)
    return tuple(stored_data[i] if stored_data[i] is not None else empty_fig for i in range(len(stored_data)))




@app.callback(
    [
        Output(f"store_plot_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_strike_price_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_plot_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] + [
        State(f"input_S0_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_K_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_T_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_r_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ],
)
def update_greek_vs_strike_price_plots(*args):
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks : n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Group states by exotic type
    n_states_per_exotic = 5  # S0, K, T, r, sigma
    split_states = [
        states[i::n_exotics] for i in range(n_exotics)
    ]

    # Initialize updated plots with empty_fig by default
    updated_plots = [
        stored_data[i] if stored_data[i] is not None else empty_fig
        for i in range(len(stored_data))
    ]

    for exotic_index, (exotic, state_set) in enumerate(zip(EXOTIC_OPTION_TYPES, split_states)):
        S0, K, T, r, sigma = state_set

        h = H
        K_range = K_RANGE  # Define a range of strike prices
        Z = Z_precomputed

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_button == f"button_compute_{greek}_vs_strike_price_{exotic}" and Z is not None:
                plot = plot_greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek)
                updated_plots[output_index] = plot

    return tuple(updated_plots)



@app.callback(
    [
        Output(f"plot_{greek}_vs_strike_price_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_plot_{greek}_vs_strike_price_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_strike_price_from_store(*stored_data):
    """
    Callback to render stored plots for Greeks vs Strike Price in their respective graphs.
    Returns empty_fig if store data is None.
    """
    return tuple(stored_data[i] if stored_data[i] is not None else empty_fig for i in range(len(stored_data)))



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
