from dash import Dash, Input, Output, html, dcc, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import dash_mantine_components as dmc
import joblib
from pricer.asian import pricer_asian, plotter_asian  # should be in ascript plotter
from pricer.lookback import pricer_lookback, plotter_lookback
from pricer.barrier import pricer_barrier, plotter_barrier
from pricer.european import pricer_european, plotter_european
from pricer.monte_carlo import monte_carlo_simulations
import plotly.graph_objects as go
from app_new_folder.components import generate_main_div, empty_fig  # Import reusable components
from app_new_folder.components_model_div import  div_models
from constants import H, S0_RANGE, K_RANGE, B_CALL, B_PUT, N_SIMULATIONS, pricer_mapping, TTM_RANGE
from greeks.delta import compute_delta
from greeks.gamma import compute_gamma
from greeks.vega import compute_vega
from greeks.theta import compute_theta
from greeks.rho import compute_rho
from greeks.greeks_functions import plot_greek_vs_stock_price, plot_greek_vs_strike_price, plot_greek_vs_ttm

# Initialize the Dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.DARKLY, "https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.9.1/font/bootstrap-icons.min.css"], )

app.title = "Price My Options"

# Load precomputed Z
Z_precomputed = joblib.load("Z_precomputed.joblib")

# Exotic options dynamically retrieved from menu_bar
EXOTIC_OPTION_TYPES = ["asian", "lookback", "barrier", "european"] 

GREEKS = ["delta", "gamma", "theta", "vega", "rho"]

# Dictionary of exotic option types and their corresponding plotters
PLOTTERS = {
    "asian": plotter_asian,
    "lookback": plotter_lookback,
    "barrier": plotter_barrier,
    "european": plotter_european,
}

# Menu bar for selecting exotic options
menu_bar = html.Div([
    dmc.SegmentedControl(
        id="menu_bar",
        value="models",
        fullWidth=True,
        data=[
            {"value": "models", "label": "Models"},
            {"value": "asian", "label": "Asian"},
            {"value": "lookback", "label": "Lookback"},
            {"value": "barrier", "label": "Barrier"},
            {"value": "european", "label": "European"},
            #{"value": "value3", "label": "Label 3"},
        ]
    )
])


# Generate divs for exotic options  #PB: during initial load, this function is called, so we can see the loading spinners
div_asian = generate_main_div("asian")
div_lookback = generate_main_div("lookback")
div_barrier = generate_main_div("barrier")
div_european = generate_main_div("european")

# Define the app layout
app.layout = html.Div([
    html.H1("Price My Options", style={"textAlign": "center", "margin-top": "20px"}),
    menu_bar,
    div_models,
    div_asian,
    div_lookback,
    div_barrier,
    div_european
], style = {'margin' : '30px'})

# Callback to toggle visibility of divs based on the menu bar selection
@app.callback(
    [Output('div_models', 'hidden'), 
     Output('div_asian', 'hidden'),
     Output('div_lookback', 'hidden'),
     Output('div_barrier', 'hidden'),
     Output('div_european', 'hidden')],
    [Input('menu_bar', 'value')],
)
def show_hidden_div(input_value):
    # Default all divs to hidden
    show_div_models = True
    show_div_asian = True
    show_div_lookback = True
    show_div_barrier = True
    show_div_european = True

    # Show only the selected div
    if input_value == 'models':
        show_div_models = False
    if input_value == 'asian':
        show_div_asian = False
    elif input_value == 'lookback':
        show_div_lookback = False
    elif input_value == 'barrier':
        show_div_barrier = False
    elif input_value == 'european':
        show_div_european = False

    return show_div_models, show_div_asian, show_div_lookback, show_div_barrier, show_div_european




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
        State(f"input_sigma_{exotic}", "value") for exotic in EXOTIC_OPTION_TYPES
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def show_plot_first_n_simulations(*args):
    """
    Callback to generate and update simulation plots for multiple exotic options.

    Parameters:
        args: A combination of n_clicks and state values dynamically passed.

    Returns:
        tuple: Figures for each exotic option type.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_clicks = args[:n_exotics]  # Button clicks for each exotic type
    states = args[n_exotics:-2]  # Exclude barrier-specific inputs (last two states)
    B_call, B_put = args[-2], args[-1]  # Barrier-specific inputs

    # Reshape states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]

    figures = []

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):
        if clicks > 0 and Z_precomputed is not None:
            S0, K, T, r, sigma = state
            Z = np.array(Z_precomputed)  # Convert Z back to NumPy array
            S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)

            if exotic == "barrier":
                if B_call is None or B_put is None:
                    figures.append(empty_fig)  # Return empty figure if barriers are missing
                    continue
                fig_call, fig_put = plotter_barrier(S, B_call, B_put, n_sim_to_plot=10)
                figures.append(fig_call)  # Append Down-and-Out Call plot
                # Uncomment below if Up-and-Out Put is in the layout
                # figures.append(fig_put)
            else:
                plotter = PLOTTERS.get(exotic, None)
                if plotter is not None:
                    fig = plotter(S, n_sim_to_plot=10)
                    fig.add_hline(
                        y=K,
                        line=dict(color="white", width=2, dash="dash"),
                        annotation_text=f"Strike Price (K={K})",
                        annotation_position="bottom right",
                    )
                    figures.append(fig)
                else:
                    figures.append(empty_fig)  # Empty figure if no plotter available
        else:
            figures.append(empty_fig)  # Empty figure if no clicks or missing Z_precomputed

    return tuple(figures)



@app.callback(
    [
        # Outputs for divs displaying Greeks
        Output(f"{greek}_{option_type}_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
        for option_type in ["call", "put"]
    ] +
    [
        # Outputs for the Greek table values
        Output(f"value_{greek}_{exotic}_call", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] +
    [
        Output(f"value_{greek}_{exotic}_put", "children")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ] +
    [
        # Outputs for the Option Prices table
        Output(f"price_call_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
    ] +
    [
        Output(f"price_put_{exotic}", "children")
        for exotic in EXOTIC_OPTION_TYPES
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
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greeks_and_prices(*args):
    """
    Callback to compute and display Greek values (Delta, Gamma, Theta, Vega, Rho)
    and Option Prices for multiple exotic options dynamically.

    Parameters:
        args: Dynamically passed inputs and states.

    Returns:
        tuple: Greek values and prices for calls and puts for all exotic options.
    """

    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)
    n_clicks = args[:n_exotics]  # Button clicks for each exotic type
    states = args[n_exotics:-2]  # Exclude barrier-specific inputs (last two states)
    B_call, B_put = args[-2], args[-1]  # Barrier-specific inputs

    # Reshape states for each exotic option type
    split_states = [states[i::n_exotics] for i in range(n_exotics)]

    div_results = []
    table_call_results = []
    table_put_results = []
    price_call_results = []
    price_put_results = []

    

    for exotic, clicks, state in zip(EXOTIC_OPTION_TYPES, n_clicks, split_states):

        if clicks > 0 and Z_precomputed is not None:
            S0, K, T, r, sigma = state
            h = H
            Z = np.array(Z_precomputed)  
            S = monte_carlo_simulations(Z, S0, T, r, sigma, n_simulations=N_SIMULATIONS)  #NOTE here we call monte_carlo simulation again, should be called once only at maximum 

            pricer = pricer_mapping.get(exotic)

            # Compute Greeks and Prices
            if exotic == "barrier":
                # Compute Greeks for barrier options
                deltas = compute_delta(Z_precomputed, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                gammas = compute_gamma(Z_precomputed, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                thetas = compute_theta(Z_precomputed, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                vegas = compute_vega(Z_precomputed, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)
                rhos = compute_rho(Z_precomputed, S0, K, T, r, sigma, h, exotic, B_call=B_call, B_put=B_put)

                # Compute Option Prices
                prices = pricer(S, K, T, r, B_call=B_call, B_put=B_put)
            else:
                # Compute Greeks for other options
                deltas = compute_delta(Z_precomputed, S0, K, T, r, sigma, h, exotic)
                gammas = compute_gamma(Z_precomputed, S0, K, T, r, sigma, h, exotic)
                thetas = compute_theta(Z_precomputed, S0, K, T, r, sigma, h, exotic)
                vegas = compute_vega(Z_precomputed, S0, K, T, r, sigma, h, exotic)
                rhos = compute_rho(Z_precomputed, S0, K, T, r, sigma, h, exotic)

                # Compute Option Prices
                prices = pricer(S, K, T, r)

            # Append results for divs
            div_results.extend([
                html.Div(f"{deltas['delta_call']:.2f}"),
                html.Div(f"{deltas['delta_put']:.2f}"),
                html.Div(f"{gammas['gamma_call']:.2f}"),
                html.Div(f"{gammas['gamma_put']:.2f}"),
                html.Div(f"{thetas['theta_call']:.2f}"),
                html.Div(f"{thetas['theta_put']:.2f}"),
                html.Div(f"{vegas['vega_call']:.2f}"),
                html.Div(f"{vegas['vega_put']:.2f}"),
                html.Div(f"{rhos['rho_call']:.2f}"),
                html.Div(f"{rhos['rho_put']:.2f}"),
            ])

            # Append results for the Greek table
            table_call_results.extend([
                f"{deltas['delta_call']:.2f}",
                f"{gammas['gamma_call']:.2f}",
                f"{thetas['theta_call']:.2f}",
                f"{vegas['vega_call']:.2f}",
                f"{rhos['rho_call']:.2f}",
            ])
            table_put_results.extend([
                f"{deltas['delta_put']:.2f}",
                f"{gammas['gamma_put']:.2f}",
                f"{thetas['theta_put']:.2f}",
                f"{vegas['vega_put']:.2f}",
                f"{rhos['rho_put']:.2f}",
            ])

            # Append results for the Prices table
            price_call_results.append(f"{prices['price_call']:.2f}")
            price_put_results.append(f"{prices['price_put']:.2f}")
        else:
            # Empty values for divs, table, and prices
            div_results.extend([html.Div('') for _ in range(n_greeks * 2)])
            table_call_results.extend(['' for _ in range(n_greeks)])
            table_put_results.extend(['' for _ in range(n_greeks)])
            price_call_results.append('')
            price_put_results.append('')

    # Combine results for divs, Greek table, and Prices table
    return tuple(div_results + table_call_results + table_put_results + price_call_results + price_put_results)





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
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_stock_price_plots(*args):
    """
    Callback to update Greek vs Stock Price plots dynamically for multiple exotic options.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Extract barrier-specific inputs
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

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
                # Handle barrier-specific logic
                if exotic == "barrier":
                    plot = plot_greek_vs_stock_price(Z, S0_range, K, T, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
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
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_strike_price_plots(*args):
    """
    Callback to dynamically update Greek vs Strike Price plots for multiple exotic options.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Extract barrier-specific inputs
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

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
                # Handle barrier-specific logic
                if exotic == "barrier":
                    plot = plot_greek_vs_strike_price(Z, S0, K_range, T, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
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


@app.callback(
    [
        Output(f"store_plot_{greek}_vs_ttm_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"button_compute_{greek}_vs_ttm_{exotic}", "n_clicks")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        State(f"store_plot_{greek}_vs_ttm_{exotic}", "data")
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
    ] + [
        State("input_B_call_barrier", "value"),
        State("input_B_put_barrier", "value"),
    ],
)
def update_greek_vs_ttm_plots(*args):
    """
    Callback to dynamically update Greek vs TTM plots for multiple exotic options.
    """
    n_exotics = len(EXOTIC_OPTION_TYPES)
    n_greeks = len(GREEKS)

    # Separate n_clicks, stored data, and states
    n_clicks = args[:n_exotics * n_greeks]
    stored_data = args[n_exotics * n_greeks: n_exotics * n_greeks * 2]
    states = args[n_exotics * n_greeks * 2:]

    # Extract barrier-specific inputs
    B_call, B_put = states[-2], states[-1]
    states = states[:-2]  # Exclude barrier inputs from the main states

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
        T_range = TTM_RANGE  # Define a range of TTM values
        Z = Z_precomputed

        for greek_index, greek in enumerate(GREEKS):
            output_index = exotic_index * n_greeks + greek_index
            triggered_button = callback_context.triggered[0]["prop_id"].split(".")[0]

            if triggered_button == f"button_compute_{greek}_vs_ttm_{exotic}" and Z is not None:
                # Handle barrier-specific logic
                if exotic == "barrier":
                    plot = plot_greek_vs_ttm(Z, S0, K, T_range, r, sigma, h, exotic, greek, B_call=B_call, B_put=B_put)
                else:
                    plot = plot_greek_vs_ttm(Z, S0, K, T_range, r, sigma, h, exotic, greek)
                updated_plots[output_index] = plot

    return tuple(updated_plots)


@app.callback(
    [
        Output(f"plot_{greek}_vs_ttm_{exotic}", "figure")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
    [
        Input(f"store_plot_{greek}_vs_ttm_{exotic}", "data")
        for exotic in EXOTIC_OPTION_TYPES
        for greek in GREEKS
    ],
)
def update_plots_greek_vs_ttm_from_store(*stored_data):
    """
    Callback to render stored plots for Greeks vs TTM in their respective graphs.
    Returns empty_fig if store data is None.
    """
    return tuple(stored_data[i] if stored_data[i] is not None else empty_fig for i in range(len(stored_data)))



# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
